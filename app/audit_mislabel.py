"""
audit_mislabel.py - Pesta Data 2026

Modul untuk mendeteksi & memperbaiki kesalahan label kategori,
berdasarkan bukti dari kolom `url` (keyword) dan `ip` (exact match + subnet /24).

Alur:
    1. Validasi format IP (invalid -> treat sebagai missing)
    2. Cek keyword url per kategori
    3. Cek dominasi kategori per IP exact & subnet /24
    4. Gabung jadi skor kecurigaan (suspect_score)
    5. Relabel baris dengan skor >= threshold

PENTING:
    - Fungsi ini HANYA dipakai di TRAINING SET (butuh kolom 'category').
    - Semua statistik (keyword match, ip dominant, subnet dominant) dihitung
      dari dataframe yang di-passing (harus train, JANGAN gabung dengan test)
      untuk menghindari data leakage.

Cara pakai di main.ipynb:
    from audit_mislabel import audit_mislabel, apply_relabel, summarize_audit, summarize_diff

    train_audited = audit_mislabel(train)
    train, diff_table = apply_relabel(train_audited, score_threshold=5)
    summarize_diff(diff_table)
"""

import re
import pandas as pd


# ============================================================
# Konfigurasi (bisa disesuaikan)
# ============================================================

KEYWORD_MAP = {
    'online gambling': ['slot', 'gacor', 'judi', 'togel', 'toto', 'poker', 'rtp', 'depo'],
    'phishing': ['verify', 'login', 'account', 'secure-', 'confirm-account'],
}

# Kosakata kategori resmi PANDI/PeDaS 2026 (evaluate.py panitia).
# PENTING: "piiexposure" TIDAK ADA underscore -- beda dari penulisan umum "pii_exposure".
# Kalau nama kategori submission tidak persis cocok salah satu dari ini,
# baris itu dianggap "category tidak dikenal" -> SELURUH submission jadi INVALID (macro_f1=None).
VALID_CATEGORIES = {
    "online gambling", "phishing", "other", "spam", "malware",
    "brand", "fakeshop", "violence", "piiexposure",
}

MIN_COUNT_IP = 5
MIN_COUNT_SUBNET = 10
MIN_COUNT_DOMAIN = 5
MIN_PURITY = 0.8

SCORE_KEYWORD = 3
SCORE_IP_EXACT = 2
SCORE_SUBNET = 2
SCORE_DOMAIN = 1


def assert_valid_categories(categories, context=""):
    """
    Wajib dipanggil sebelum menyimpan submission atau setelah cat_map diterapkan.
    Cegah bug fatal: nama kategori yang tidak persis cocok VALID_CATEGORIES
    akan membuat SELURUH submission dianggap invalid oleh evaluator panitia.
    """
    unexpected = set(categories) - VALID_CATEGORIES
    if unexpected:
        raise ValueError(
            f"Kategori tidak dikenal evaluator ({context}): {unexpected}\n"
            f"Kategori resmi yang valid: {sorted(VALID_CATEGORIES)}"
        )


# ============================================================
# Langkah 0: Validasi format IP
# ============================================================
def is_valid_ipv4(ip):
    """True kalau valid, False kalau invalid, None kalau missing."""
    if pd.isna(ip):
        return None
    m = re.match(r'^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$', str(ip).strip())
    if not m:
        return False
    return all(0 <= int(x) <= 255 for x in m.groups())


def _build_dominance_stats(df, groupcol, category_col='category'):
    """Hitung kategori dominan + purity + jumlah baris per grup (ip/subnet/domain)."""
    stats = df.dropna(subset=[groupcol]).groupby(groupcol)[category_col].agg(
        dominant=lambda x: x.value_counts().idxmax(),
        purity=lambda x: x.value_counts().max() / len(x),
        count='count'
    )
    return stats


# ============================================================
# Fungsi utama: audit
# ============================================================
def audit_mislabel(df, category_col='category', url_col='url', ip_col='ip', domain_col='domain'):
    """
    Audit kemungkinan salah label. Return dataframe asli + kolom tambahan:
        - ip_valid, ip_prefix24        : hasil validasi & breakdown subnet
        - kw_<kategori>                : True/False url mengandung keyword kategori tsb
        - ip_dominant/purity/count     : statistik dominasi per IP exact
        - subnet_dominant/purity/count : statistik dominasi per subnet /24
        - domain_dominant/purity/count : statistik dominasi per domain
        - suspect_score                : skor kecurigaan (makin tinggi makin yakin salah label)
        - suspect_reasons              : daftar alasan (list of string)
    """
    df = df.copy()

    # ---------- Langkah 0: validasi IP ----------
    df['ip_valid'] = df[ip_col].apply(is_valid_ipv4)
    df.loc[df['ip_valid'] == False, ip_col] = None

    # ---------- Langkah 1: cek url (keyword per kategori) ----------
    for cat, kws in KEYWORD_MAP.items():
        col = f'kw_{cat.replace(" ", "_")}'
        df[col] = df[url_col].str.lower().str.contains('|'.join(kws), na=False)

    # ---------- Langkah 2: cek ip (exact + subnet) + domain sbg pelengkap ----------
    df['ip_prefix24'] = df[ip_col].str.rsplit('.', n=1).str[0]

    ip_stats = _build_dominance_stats(df, ip_col, category_col).add_prefix('ip_')
    subnet_stats = _build_dominance_stats(df, 'ip_prefix24', category_col).add_prefix('subnet_')
    domain_stats = _build_dominance_stats(df, domain_col, category_col).add_prefix('domain_')

    df = df.merge(ip_stats, left_on=ip_col, right_index=True, how='left')
    df = df.merge(subnet_stats, left_on='ip_prefix24', right_index=True, how='left')
    df = df.merge(domain_stats, left_on=domain_col, right_index=True, how='left')

    # ---------- Langkah 3: gabung jadi skor ----------
    def compute_score(row):
        score = 0
        reasons = []

        for cat in KEYWORD_MAP:
            col = f'kw_{cat.replace(" ", "_")}'
            if row[col] and row[category_col] != cat:
                score += SCORE_KEYWORD
                reasons.append(f"keyword_url->{cat}(+{SCORE_KEYWORD})")

        if (pd.notna(row['ip_dominant']) and row['ip_count'] >= MIN_COUNT_IP
                and row['ip_purity'] >= MIN_PURITY and row[category_col] != row['ip_dominant']):
            score += SCORE_IP_EXACT
            reasons.append(f"ip_exact->{row['ip_dominant']}(+{SCORE_IP_EXACT})")

        if (pd.notna(row['subnet_dominant']) and row['subnet_count'] >= MIN_COUNT_SUBNET
                and row['subnet_purity'] >= MIN_PURITY and row[category_col] != row['subnet_dominant']):
            score += SCORE_SUBNET
            reasons.append(f"subnet->{row['subnet_dominant']}(+{SCORE_SUBNET})")

        if (pd.notna(row['domain_dominant']) and row['domain_count'] >= MIN_COUNT_DOMAIN
                and row['domain_purity'] >= MIN_PURITY and row[category_col] != row['domain_dominant']):
            score += SCORE_DOMAIN
            reasons.append(f"domain->{row['domain_dominant']}(+{SCORE_DOMAIN})")

        return pd.Series([score, reasons])

    df[['suspect_score', 'suspect_reasons']] = df.apply(compute_score, axis=1)

    return df


def get_reference_stats(df_train, ip_col='ip', category_col='category'):
    """
    Ambil ip_stats & subnet_stats dari TRAIN, untuk dipakai ulang
    di score_confidence_evidence() saat memproses TEST (yang gak punya category).
    """
    df = df_train.copy()
    df['ip_valid'] = df[ip_col].apply(is_valid_ipv4)
    df.loc[df['ip_valid'] == False, ip_col] = None
    df['ip_prefix24'] = df[ip_col].str.rsplit('.', n=1).str[0]

    ip_stats = _build_dominance_stats(df, ip_col, category_col)
    subnet_stats = _build_dominance_stats(df, 'ip_prefix24', category_col)
    return ip_stats, subnet_stats


def summarize_audit(df_audited, score_col='suspect_score'):
    """Cetak ringkasan distribusi skor kecurigaan."""
    print("=== Distribusi skor kecurigaan ===")
    print(df_audited[score_col].value_counts().sort_index(ascending=False))
    print(f"\nSkor >=5 (relabel otomatis)      : {(df_audited[score_col] >= 5).sum()} baris")
    print(f"Skor 3-4 (verifikasi manual)     : {((df_audited[score_col] >= 3) & (df_audited[score_col] < 5)).sum()} baris")
    print(f"Skor 1-2 (biarkan, terlalu lemah): {((df_audited[score_col] >= 1) & (df_audited[score_col] < 3)).sum()} baris")


# ============================================================
# Fungsi relabel
# ============================================================
def _pick_relabel_target(row):
    reasons = row['suspect_reasons']
    if not reasons:
        return row['category']
    targets = [r.split('->')[1].split('(')[0] for r in reasons if '->' in r]
    if not targets:
        return row['category']
    return pd.Series(targets).value_counts().idxmax()


def _explain_reasons(reasons):
    if not reasons:
        return ""
    label_map = {
        'keyword_url': "URL mengandung kata kunci khas '{target}'",
        'ip_exact': "IP yang sama persis sudah terbukti dipakai untuk '{target}' di baris lain",
        'subnet': "IP berada di rentang/subnet yang didominasi kategori '{target}'",
        'domain': "Domain yang sama sudah terbukti dominan berkategori '{target}'",
    }
    explanations = []
    for r in reasons:
        code, rest = r.split('->')
        target, poin = rest.split('(')
        poin = poin.rstrip(')')
        template = label_map.get(code, code)
        explanations.append(f"{template.format(target=target)} [{poin}]")
    return " ; ".join(explanations)


def apply_relabel(df_audited, score_threshold=5, category_col='category'):
    """
    Terapkan relabel untuk baris dengan suspect_score >= score_threshold.

    Return: (df_relabeled, diff_table)
    """
    df = df_audited.copy()
    df['category_before'] = df[category_col]

    mask = df['suspect_score'] >= score_threshold
    df.loc[mask, category_col] = df.loc[mask].apply(_pick_relabel_target, axis=1)

    changed_mask = df[category_col] != df['category_before']
    diff_table = df.loc[changed_mask, [
        'url', 'category_before', category_col, 'suspect_score', 'suspect_reasons'
    ]].rename(columns={category_col: 'category_after'})
    diff_table['keterangan'] = diff_table['suspect_reasons'].apply(_explain_reasons)

    return df, diff_table


def summarize_diff(diff_table):
    """Cetak ringkasan perubahan label."""
    print(f"=== Total baris yang benar-benar berubah label: {len(diff_table)} ===\n")
    if len(diff_table) == 0:
        print("Tidak ada perubahan.")
        return
    print("=== Perubahan per pasangan (label lama -> label baru) ===")
    print(diff_table.groupby(['category_before', 'category_after']).size().sort_values(ascending=False))


# ============================================================
# Fungsi tambahan: perbaiki confidence_level yang anomali
# pakai bukti yang SAMA dari audit_mislabel (reuse, bukan hitung ulang)
# ============================================================
def recompute_confidence(df_audited, confidence_col='confidence_level'):
    """
    Untuk baris dengan confidence_level di luar 0-100, hitung ulang
    pakai bukti url+ip yang sama dari audit_mislabel (kw_*, ip_purity, dst).
    Baris yang confidence-nya sudah valid (0-100) tidak diubah.

    CATATAN: fungsi ini bisa dipakai untuk TRAIN maupun TEST, selama
    df_audited sudah melalui score_confidence_evidence() (bukan audit_mislabel()
    langsung, karena test gak punya kolom 'category').
    """
    df = df_audited.copy()

    def _recompute(row):
        val = row[confidence_col]
        if 0 <= val <= 100:
            return val
        score = 0
        if any(row.get(f'kw_{cat.replace(" ", "_")}', False) for cat in KEYWORD_MAP):
            score += 60
        if pd.notna(row.get('ip_purity')) and row.get('ip_count', 0) >= MIN_COUNT_IP and row['ip_purity'] >= MIN_PURITY:
            score += 40
        elif pd.notna(row.get('subnet_purity')) and row.get('subnet_count', 0) >= MIN_COUNT_SUBNET and row['subnet_purity'] >= MIN_PURITY:
            score += 25
        return min(score, 100)

    df[confidence_col] = df.apply(_recompute, axis=1)
    return df


def score_confidence_evidence(df, ref_ip_stats, ref_subnet_stats, url_col='url', ip_col='ip'):
    """
    Versi generalisasi untuk TEST (tidak butuh kolom 'category').
    Terapkan referensi statistik (ip_stats, subnet_stats) yang SUDAH DIHITUNG
    dari TRAIN, ke dataframe manapun (train atau test) -- tanpa menghitung ulang.

    ref_ip_stats, ref_subnet_stats: hasil _build_dominance_stats() dari TRAIN saja.
    """
    df = df.copy()
    df['ip_valid'] = df[ip_col].apply(is_valid_ipv4)
    df.loc[df['ip_valid'] == False, ip_col] = None
    df['ip_prefix24'] = df[ip_col].str.rsplit('.', n=1).str[0]

    for cat, kws in KEYWORD_MAP.items():
        col = f'kw_{cat.replace(" ", "_")}'
        df[col] = df[url_col].str.lower().str.contains('|'.join(kws), na=False)

    df = df.merge(ref_ip_stats.add_prefix('ip_'), left_on=ip_col, right_index=True, how='left')
    df = df.merge(ref_subnet_stats.add_prefix('subnet_'), left_on='ip_prefix24', right_index=True, how='left')

    return df


if __name__ == "__main__":
    print("Modul ini dirancang untuk di-IMPORT dari notebook, bukan dijalankan langsung.")
    print("Contoh pemakaian:")
    print("  from audit_mislabel import audit_mislabel, apply_relabel, assert_valid_categories")
    print("  train_audited = audit_mislabel(train)")
    print("  assert_valid_categories(train['category'].unique(), context='train setelah cleaning')")
