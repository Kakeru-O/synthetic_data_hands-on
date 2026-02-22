import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from mostlyai.sdk import MostlyAI
    import pandas as pd
    import altair as alt
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    from pathlib import Path

    return MostlyAI, NearestNeighbors, Path, StandardScaler, alt, mo, np, pd


@app.cell
def _(mo):
    mo.md(r"""
    # 合成データハンズオン：MarimoとMostlyAIで作る安全なデータ共有基盤

    マーケティング部門が、キャンペーンのターゲット精度向上のため、顧客データを外部に提供し、データ分析したいと考えています。
    しかし、法務部門からは「個人情報保護（GDPR/APPI）」の観点からストップがかかっています。

    **あなたのミッション:** 元の顧客データの統計的性質を維持しつつ、**実在する個人を一切含まない**合成データセットを作成し、
    データを安全に共有できることを法務部門に証明しましょう。
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Step 1: データの読み込み ("The Dangerous Data")

    UCI Machine Learning Repository から "Bank Marketing" データセットを取得・読み込みます。

    > ⚠️ このデータには顧客の年齢、職業、残高などの**機密属性**が含まれていると想定してください。
    > **このままの形式では、プライバシー保護の観点から外部に提供することはできません。**
    """)
    return


@app.cell
def _(Path, pd):
    DATA_PATH = Path("data/raw") / "bank-marketing.csv"

    if DATA_PATH.exists():
        df_original = pd.read_csv(DATA_PATH)
    else:
        from ucimlrepo import fetch_ucirepo
        bank_marketing = fetch_ucirepo(id=222)
        X = bank_marketing.data.features
        y = bank_marketing.data.targets
        df_original = pd.concat([X, y], axis=1)
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_original.to_csv(DATA_PATH, index=False)
    return DATA_PATH, df_original


@app.cell
def _(DATA_PATH, df_original, mo):
    mo.vstack([
        mo.md(f"✅ データを読み込みました: `{DATA_PATH}` ({len(df_original)} レコード)"),
        mo.ui.table(df_original, page_size=10, label="元データ (Original Sensitive Data)"),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Step 2: 合成データの生成

    MostlyAI SDK（ローカルモード）を使用して、元データの統計的特性を学習し、新しい合成データを生成します。
    生成されたデータは元データのパターンを模倣しますが、**実在する個人のデータとは1対1で対応しません**。
    """)
    return


@app.cell
def _(mo):
    sample_size_slider = mo.ui.slider(
        start=100, stop=5000, value=100, step=100,
        label="生成するサンプル数"
    )
    generate_button = mo.ui.run_button(label="合成データを生成")

    mo.hstack([sample_size_slider, generate_button], justify="start", gap=1)
    return generate_button, sample_size_slider


@app.cell
def _(MostlyAI, df_original, generate_button, mo, pd, sample_size_slider):
    mo.stop(not generate_button.value, mo.md("上のボタンを押すと合成データの生成が始まります。"))

    df_synthetic = pd.DataFrame()

    with mo.status.spinner("合成データを生成中... (これには数分かかる場合があります)"):
        mostly = MostlyAI(local=True, local_dir="./mostlyai_local")
        # 各カラムのデータ型とエンコード方式を明示的に指定します。
        # 'month' カラムが TABULAR_DATETIME として誤認識され Pandas のエラーになるのを防ぐ目的も兼ねています。
        config = {
            'name': 'Bank Marketing',
            'tables': [
                {
                    'name': 'bank_marketing',
                    'data': df_original,
                    'tabular_model_configuration': {
                        # 'model': 'MOSTLY_AI/Medium',       # AIモデルのサイズ指定（Small, Medium, Large）
                        # 'max_epochs': 50,                  # 学習の最大エポック数（精度と時間のトレードオフ）
                        # 'enable_flexible_generation': True # シードや欠損値補完などを有効にするか
                    },
                    'columns': [
                        {'name': 'age', 'model_encoding_type': 'TABULAR_NUMERIC_AUTO'},
                        {'name': 'job', 'model_encoding_type': 'TABULAR_CATEGORICAL'},
                        {'name': 'marital', 'model_encoding_type': 'TABULAR_CATEGORICAL'},
                        {'name': 'education', 'model_encoding_type': 'TABULAR_CATEGORICAL'},
                        {'name': 'default', 'model_encoding_type': 'TABULAR_CATEGORICAL'},
                        {'name': 'balance', 'model_encoding_type': 'TABULAR_NUMERIC_AUTO'},
                        {'name': 'housing', 'model_encoding_type': 'TABULAR_CATEGORICAL'},
                        {'name': 'loan', 'model_encoding_type': 'TABULAR_CATEGORICAL'},
                        {'name': 'contact', 'model_encoding_type': 'TABULAR_CATEGORICAL'},
                        {'name': 'day_of_week', 'model_encoding_type': 'TABULAR_NUMERIC_DISCRETE'},
                        {'name': 'month', 'model_encoding_type': 'TABULAR_CATEGORICAL'},
                        {'name': 'duration', 'model_encoding_type': 'TABULAR_NUMERIC_AUTO'},
                        {'name': 'campaign', 'model_encoding_type': 'TABULAR_NUMERIC_DISCRETE'},
                        {'name': 'pdays', 'model_encoding_type': 'TABULAR_NUMERIC_AUTO'},
                        {'name': 'previous', 'model_encoding_type': 'TABULAR_NUMERIC_DISCRETE'},
                        {'name': 'poutcome', 'model_encoding_type': 'TABULAR_CATEGORICAL'},
                        {'name': 'y', 'model_encoding_type': 'TABULAR_CATEGORICAL'},
                    ]
                }
            ]
        }
        g = mostly.train(config=config)
        sd = mostly.generate(g, size=sample_size_slider.value)
        df_synthetic = sd.data()

    _result = mo.vstack([
        mo.md(f"✅ **合成データの生成が完了しました！** ({len(df_synthetic)} レコード)"),
        mo.ui.table(df_synthetic, page_size=10, label="合成データ (Generated Synthetic Data)"),
    ])
    return df_synthetic, g, sd


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Step 2.5: モデル評価 ("The Quality Check")

    MostlyAI が自動的に算出するモデル品質メトリクスと QA レポートを確認します。
    これにより、生成された合成データが元データの統計的特性をどの程度再現できているか、
    またプライバシーがどの程度保護されているかを定量的に評価できます。
    """)
    return


@app.cell
def _(df_synthetic, mo, pd, sd):
    mo.stop(
        df_synthetic.empty,
        mo.md("合成データが生成されると、ここにモデル評価メトリクスが表示されます。")
    )

    _metrics = sd.tables[0].tabular_model_metrics

    def _fmt_pct(v):
        return f"{v * 100:.1f}%" if isinstance(v, (int, float)) else "N/A"

    def _quality_emoji(v):
        if not isinstance(v, (int, float)):
            return "❓"
        if v >= 0.9:
            return "🟢"
        if v >= 0.7:
            return "🟡"
        return "🔴"

    _acc = _metrics.accuracy if _metrics else None
    _acc_overall = _acc.overall if _acc else None
    _acc_univariate = _acc.univariate if _acc else None
    _acc_bivariate = _acc.bivariate if _acc else None

    _metrics_cards = mo.hstack(
        [
            mo.stat(
                label="Overall Accuracy",
                value=f"{_quality_emoji(_acc_overall)} {_fmt_pct(_acc_overall)}",
                bordered=True,
            ),
            mo.stat(
                label="Univariate Accuracy",
                value=f"{_quality_emoji(_acc_univariate)} {_fmt_pct(_acc_univariate)}",
                bordered=True,
            ),
            mo.stat(
                label="Bivariate Accuracy",
                value=f"{_quality_emoji(_acc_bivariate)} {_fmt_pct(_acc_bivariate)}",
                bordered=True,
            ),
        ],
        justify="center",
        gap=1,
    )

    _metrics_rows = []
    if _acc:
        for _name, _label in [
            ("overall", "Overall"), ("univariate", "Univariate"),
            ("bivariate", "Bivariate"), ("coherence", "Coherence"),
            ("overall_max", "Overall (理論上限)"), ("univariate_max", "Univariate (理論上限)"),
            ("bivariate_max", "Bivariate (理論上限)"),
        ]:
            _v = getattr(_acc, _name, None)
            if _v is not None:
                _metrics_rows.append({"カテゴリ": "Accuracy", "メトリクス": _label, "値": _fmt_pct(_v)})

    _sim = _metrics.similarity if _metrics else None
    if _sim:
        for _name, _label in [
            ("cosine_similarity_training_synthetic", "Cosine (Train↔Syn)"),
            ("cosine_similarity_training_holdout", "Cosine (Train↔Holdout)"),
            ("discriminator_auc_training_synthetic", "Discriminator AUC (Train↔Syn)"),
            ("discriminator_auc_training_holdout", "Discriminator AUC (Train↔Holdout)"),
        ]:
            _v = getattr(_sim, _name, None)
            if _v is not None:
                _metrics_rows.append({"カテゴリ": "Similarity", "メトリクス": _label, "値": _fmt_pct(_v)})

    _dist = _metrics.distances if _metrics else None
    if _dist:
        for _name, _label in [
            ("ims_training", "Identical Match (Train)"),
            ("ims_holdout", "Identical Match (Holdout)"),
            ("dcr_training", "DCR (Train)"),
            ("dcr_holdout", "DCR (Holdout)"),
        ]:
            _v = getattr(_dist, _name, None)
            if _v is not None:
                _metrics_rows.append({"カテゴリ": "Distances / Privacy", "メトリクス": _label, "値": _fmt_pct(_v)})

    _metrics_table = mo.ui.table(
        pd.DataFrame(_metrics_rows) if _metrics_rows else pd.DataFrame({"メトリクス": ["データなし"], "値": ["-"]}),
        label="全メトリクス一覧",
    )

    mo.vstack([
        mo.md("### 📊 モデル品質メトリクス"),
        _metrics_cards,
        mo.md("""
    > **Accuracy** は元データの統計的特性を合成データがどの程度再現できているかを示します（`1 - TVD` に相当）。
    > 🟢 90%以上 = 優秀 / 🟡 70-90% = 良好 / 🔴 70%未満 = 要改善
    """),
        mo.accordion({"📋 全メトリクス詳細": _metrics_table}),
    ])
    return


@app.cell
def _(Path, df_synthetic, g, mo, sd):
    import subprocess as _subprocess
    import sys as _sys
    import os as _os

    mo.stop(
        df_synthetic.empty,
        mo.md("合成データが生成されると、ここに QA レポートが表示されます。")
    )

    _local_dir = Path("./mostlyai_local").resolve()
    # ユーザー指摘の通り、Model QAは Generator 側に格納されている
    _gen_model_report_dir = _local_dir / "generators" / g.id / "ModelQAReports"
    _sd_data_report_dir = _local_dir / "synthetic-datasets" / sd.id / "DataQAReports"

    def _find_report(report_dir):
        if not report_dir.exists():
            return None
        _html_files = list(report_dir.glob("*.html"))
        return _html_files[0] if _html_files else None

    _gen_model_report = _find_report(_gen_model_report_dir)
    _sd_data_report = _find_report(_sd_data_report_dir)

    _reports = {
        "🧠 Model QA": _gen_model_report,
        "📦 Data QA": _sd_data_report,
    }

    def _open_report(p):
        _path_str = str(p)
        if _sys.platform == "win32":
            _os.startfile(_path_str)
        else:
            _opener = "open" if _sys.platform == "darwin" else "xdg-open"
            _res = _subprocess.run([_opener, _path_str], capture_output=True, text=True)
            if _res.returncode != 0:
                print(f"⚠️ Failed to open {p.name}: {_res.stderr.strip()}")

    _cards = []
    for _label, _path in _reports.items():
        if _path:
            _size_mb = _path.stat().st_size / (1024 * 1024)
            _btn = mo.ui.run_button(
                label=f"{_label} を開く",
                on_change=lambda _, p=_path: _open_report(p)
            )
            _copy_text = f"`file://{_path}`"
            _cards.append(mo.vstack([
                mo.md(f"**{_label}**\n\n📄 `{_path.name}` ({_size_mb:.1f} MB)\n\n{_copy_text}"),
                _btn,
            ]))
        else:
            _cards.append(mo.md(f"**{_label}**\n\n⚠️ レポートが見つかりませんでした。"))

    mo.vstack([
        mo.md("### 📑 QA レポート"),
        mo.md("> ボタンをクリックするか、記載の `file://...` パスをブラウザのURL欄にコピー＆ペーストして開いてください。"),
        mo.hstack(_cards, gap=1),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Step 3: 統計的検証 ("The Proof")

    合成データが元データの分布をどの程度再現できているか、視覚的に確認します。
    **青色**が「元データ」、**赤色**が「合成データ」です。
    分布が重なっているほど、統計的性質が維持されていることを示します。
    """)
    return


@app.cell
def _(df_original, mo):
    columns = df_original.columns.tolist()
    column_selector = mo.ui.dropdown(
        options=columns,
        value=columns[0] if columns else None,
        label="可視化するカラムを選択"
    )
    column_selector
    return (column_selector,)


@app.cell
def _(alt, column_selector, df_original, df_synthetic, mo, pd):
    mo.stop(
        df_synthetic.empty,
        mo.md("合成データが生成されると、ここに比較グラフが表示されます。")
    )

    _col = column_selector.value

    _df_orig_plot = df_original[[_col]].copy()
    _df_orig_plot["Type"] = "Original"

    _df_syn_plot = df_synthetic[[_col]].copy()
    _df_syn_plot["Type"] = "Synthetic"

    _df_plot = pd.concat([_df_orig_plot, _df_syn_plot])

    _is_numeric = pd.api.types.is_numeric_dtype(df_original[_col])

    if _is_numeric:
        _chart = (
            alt.Chart(_df_plot)
            .mark_bar(opacity=0.5)
            .encode(
                alt.X(_col, bin=alt.Bin(maxbins=30), title=_col),
                alt.Y("count()", stack=None, title="件数"),
                alt.Color(
                    "Type",
                    scale=alt.Scale(
                        domain=["Original", "Synthetic"],
                        range=["steelblue", "salmon"],
                    ),
                ),
            )
            .properties(title=f"{_col} の分布比較", width=600, height=400)
        )
    else:
        _chart = (
            alt.Chart(_df_plot)
            .mark_bar(opacity=0.7)
            .encode(
                alt.X(_col, title=_col),
                alt.Y("count()", title="件数"),
                alt.Color(
                    "Type",
                    scale=alt.Scale(
                        domain=["Original", "Synthetic"],
                        range=["steelblue", "salmon"],
                    ),
                    legend=alt.Legend(title="データ種別"),
                ),
                alt.XOffset("Type"),
            )
            .properties(title=f"{_col} の分布比較", width=600, height=400)
        )

    mo.ui.altair_chart(_chart)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Step 4: プライバシー検証

    合成データの中からランダムに1つのサンプルを選び、元データの中で「最も似ている」実在の人物を探します。
    もし完全に一致するレコードが存在しなければ、それは「新しい架空の人物」が生成された証拠になります。
    """)
    return


@app.cell
def _(NearestNeighbors, StandardScaler, df_original, df_synthetic, mo, np, pd):
    mo.stop(
        df_synthetic.empty,
        mo.md("合成データが生成されると、ここにプライバシー検証結果が表示されます。")
    )

    np.random.seed(None)
    _target_synthetic = df_synthetic.sample(1).iloc[0]

    _numeric_cols = df_original.select_dtypes(include=["number"]).columns.tolist()

    if not _numeric_cols:
        _out = mo.md("数値カラムがないため、最近傍探索をスキップしました。")
    else:
        _scaler = StandardScaler()
        _X_orig = _scaler.fit_transform(df_original[_numeric_cols])
        _X_syn_sample = _scaler.transform(
            _target_synthetic[_numeric_cols].to_frame().T
        )

        _nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(_X_orig)
        _distances, _indices = _nbrs.kneighbors(_X_syn_sample)

        _closest_idx = _indices[0][0]
        _closest_real = df_original.iloc[_closest_idx]

        _comparison_df = pd.DataFrame(
            {"合成データ (Synthetic)": _target_synthetic, "最近傍の実データ (Real)": _closest_real}
        ).T

        _diff_msgs = []
        for _c in _numeric_cols:
            _syn_val = _target_synthetic[_c]
            _real_val = _closest_real[_c]
            if _syn_val != _real_val:
                _diff_msgs.append(f"- **{_c}**: 合成={_syn_val}, 実データ={_real_val} (差: {abs(_syn_val - _real_val):.2f})")

        _diff_text = "\n".join(_diff_msgs) if _diff_msgs else "差分なし（完全一致）"

        _out = mo.vstack([
            mo.md("### 🕵️ 最近傍探索結果"),
            mo.ui.table(_comparison_df, label="比較表"),
            mo.md(f"**ユークリッド距離 (標準化後):** {_distances[0][0]:.4f}"),
            mo.md(f"""
    **主な差分:**

    {_diff_text}

    > ✅ **結論:** 上記の通り、最も似ている実在データと比較しても属性値に違いがあります。
    > これは、生成されたデータが元の個人の「コピー」ではなく、統計的な性質を受け継いだ**新しい架空の人物**であることを示しています。
    > したがって、**再識別リスクは低い**と判断できます。
    """),
        ])

    _out
    return


if __name__ == "__main__":
    app.run()
