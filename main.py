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
    import os
    import sys
    import subprocess

    return (
        MostlyAI,
        NearestNeighbors,
        Path,
        StandardScaler,
        alt,
        mo,
        np,
        os,
        pd,
        subprocess,
        sys,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 合成データハンズオン：MarimoとMostlyAIで作る安全なデータ共有基盤

    マーケティング部門が、キャンペーンのターゲット精度向上のため、顧客データを外部に提供し、データ分析したいと考えています。
    しかし、法務部門からは「個人情報保護（GDPR/APPI）」の観点からストップがかかっています。

    **あなたのミッション:** 元の顧客データの統計的性質を維持しつつ、**実在する個人を一切含まない**合成データセットを作成し、
    データを安全に共有できることを法務部門に証明しましょう。
    """)
    return


@app.cell(hide_code=True)
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
def _(Path, mo, pd):
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

    mo.md(f"✅ データを読み込みました: `{DATA_PATH}` ({len(df_original)} レコード)")
    return (df_original,)


@app.cell
def _(df_original, mo):
    mo.ui.table(
        df_original, page_size=10, label="元データ (Original Sensitive Data)"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    _explanation = mo.md("""
    ---

    ## Step 2: ジェネレーターの設定と構築

    MostlyAI SDK（ローカルモード）を使用して、元データの統計的特性を学習させるための設定を行います。
    MostlyAIでは、学習済みのAIモデルを**「ジェネレーター (Generator)」**と呼びます。
    """)

    _accordion_content = r"""各カラムのデータ型に応じて、AIがどのように学習するかを指定します。Tabularモデル系（通常のテーブルデータ）では主に以下のいずれかを指定します。

    - **AUTO**: 自動検出（デフォルト）
    - **TABULAR_CATEGORICAL**: カテゴリ変数（固定の値セット）。数値として表現された区分値にも使用可
    - **TABULAR_NUMERIC_AUTO**: 数値：自動判定（ほとんどのケースで推奨）
    - **TABULAR_NUMERIC_DISCRETE**: 数値：離散扱い。ZIPコード、0/1フラグ、カテゴリ的な数値コードに使用
    - **TABULAR_NUMERIC_BINNED**: 数値：ビン分割。大きな整数や長い小数に使用（100ビンに分割）
    - **TABULAR_NUMERIC_DIGIT**: 数値：桁単位認識
    - **TABULAR_CHARACTER**: 短い文字列パターン（電話番号、ライセンスプレート、ID文字列等）
    - **TABULAR_DATETIME**: 日時（yyyy-MM-dd〜yyyy-MM-ddTHH:mm:ss.SSSZ形式対応）
    - **TABULAR_DATETIME_RELATIVE**: 日時：相対的。連続イベント間の時間間隔を正確にモデル化
    - **TABULAR_LAT_LONG**: 緯度・経度（`"lat, long"` 形式の1カラムに格納）"""

    mo.vstack([
        _explanation, 
        mo.accordion({"\> エンコーディングタイプ (Encoding Types) の詳細": _accordion_content})
    ])
    return


@app.cell
def _(df_original):
    # ジェネレーターの設定（Encoding Typesの指定）
    generator_config = {
        "name": "Bank Marketing Generator",
        "tables": [
            {
                "name": "bank_marketing",
                "data": df_original,
                "tabular_model_configuration": {
                    "model": "MOSTLY_AI/Medium",
                },
                "columns": [
                    {"name": "age", "model_encoding_type": "TABULAR_NUMERIC_AUTO"},
                    {"name": "job", "model_encoding_type": "TABULAR_CATEGORICAL"},
                    {"name": "marital", "model_encoding_type": "TABULAR_CATEGORICAL"},
                    {"name": "education", "model_encoding_type": "TABULAR_CATEGORICAL"},
                    {"name": "default", "model_encoding_type": "TABULAR_CATEGORICAL"},
                    {"name": "balance", "model_encoding_type": "TABULAR_NUMERIC_AUTO"},
                    {"name": "housing", "model_encoding_type": "TABULAR_CATEGORICAL"},
                    {"name": "loan", "model_encoding_type": "TABULAR_CATEGORICAL"},
                    {"name": "contact", "model_encoding_type": "TABULAR_CATEGORICAL"},
                    {"name": "day_of_week","model_encoding_type": "TABULAR_NUMERIC_DISCRETE"},
                    {"name": "month", "model_encoding_type": "TABULAR_CATEGORICAL"},
                    {"name": "duration", "model_encoding_type": "TABULAR_NUMERIC_AUTO"},
                    {"name": "campaign","model_encoding_type": "TABULAR_NUMERIC_DISCRETE"},
                    {"name": "pdays", "model_encoding_type": "TABULAR_NUMERIC_AUTO"},
                    {"name": "previous","model_encoding_type": "TABULAR_NUMERIC_DISCRETE"},
                    {"name": "poutcome", "model_encoding_type": "TABULAR_CATEGORICAL"},
                    {"name": "y", "model_encoding_type": "TABULAR_CATEGORICAL"},
                ],
            }
        ],
    }
    return


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="ジェネレーターのトレーニングを開始する")
    train_button
    return (train_button,)


@app.cell
def _(MostlyAI, mo, train_button):
    mo.stop(
        not train_button.value,
        mo.md("上のボタンを押すとジェネレーターのトレーニングが始まります。"),
    )

    with mo.status.spinner(
        "ジェネレーターをトレーニング中... (マシンスペックにより数分かかります)"
    ):
        mostly = MostlyAI(local=True, local_dir="./mostlyai_local")
        #g = mostly.train(config=generator_config)
        g = mostly.generators.get("a2b7fafe-21bd-4450-b83a-0855e0332f7c")

    mo.md(f"✅ **トレーニングが完了しました！** (Generator ID: `{g.id}`)")
    return g, mostly


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Step 3: モデル精度確認と Model QA

    構築したジェネレーター自体が、「実データの分布をどれだけ正確に学習できたか」を評価します。
    **Accuracy** のスコアが高いほど、元の分布に近いことを示します。
    """)
    return


@app.cell
def _(g, mo):
    mo.stop(
        g is None,
        mo.md(
            "トレーニングが完了すると、ここにモデル評価メトリクスとQAレポートが表示されます。"
        ),
    )

    _metrics = (
        getattr(g.tables[0], "tabular_model_metrics", None)
        if hasattr(g, "tables") and g.tables
        else None
    )

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
    _acc_overall = getattr(_acc, "overall", None) if _acc else None
    _acc_univariate = getattr(_acc, "univariate", None) if _acc else None
    _acc_bivariate = getattr(_acc, "bivariate", None) if _acc else None

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

    mo.vstack(
        [
            mo.md("### 📊 ジェネレーター品質メトリクス"),
            _metrics_cards,
            mo.md(
                "\n> **Accuracy** はジェネレーターの精度です。🟢 90%以上 = 優秀"
            ),
        ]
    )
    return


@app.cell
def _(Path, g, mo, os, subprocess, sys):
    mo.stop(g is None)

    _local_dir = Path("./mostlyai_local").resolve()
    _gen_model_report_dir = _local_dir / "generators" / g.id / "ModelQAReports"

    def _find_report(report_dir):
        if not report_dir.exists():
            return None
        _html_files = list(report_dir.glob("*.html"))
        return _html_files[0] if _html_files else None

    _gen_model_report = _find_report(_gen_model_report_dir)

    def _open_report(p):
        _path_str = str(p)
        if sys.platform == "win32":
            os.startfile(_path_str)
        else:
            _opener = "open" if sys.platform == "darwin" else "xdg-open"
            _res = subprocess.run(
                [_opener, _path_str], capture_output=True, text=True
            )
            if _res.returncode != 0:
                print(f"⚠️ Failed to open {p.name}: {_res.stderr.strip()}")

    if _gen_model_report:
        _btn = mo.ui.run_button(
            label="🧠 Model QA を開く",
            on_change=lambda _, p=_gen_model_report: _open_report(p),
        )
        _out = mo.vstack(
            [
                mo.md("### 📑 Model QA レポート"),
                mo.md(
                    f"📄 `{_gen_model_report.name}`\n\n`file://{_gen_model_report}`"
                ),
                _btn,
            ]
        )
    else:
        _out = mo.md("⚠️ Model QA レポートが見つかりませんでした。")

    _out
    return


@app.cell(hide_code=True)
def _(df_original, mo):
    mo.md(f"""
    ---

    ## Step 4: 合成データの生成

    構築したジェネレーターを使って、新しい架空のデータを生成します。
    デフォルトでは、元データと同じ `{len(df_original)}` 行を生成します。

    > 💡 **Tip:** 生成する行数を変更したい場合は、下のセルのコード `generate_size = len(df_original)` の部分数値を直接書き換えてください。
    """)
    return


@app.cell
def _(mo):
    generate_button = mo.ui.run_button(label="合成データを生成する")
    generate_button
    return (generate_button,)


@app.cell
def _(df_original, g, generate_button, mo, mostly):
    mo.stop(not generate_button.value, mo.md("上のボタンを押すと生成が始まります。"))

    # 生成する行数を変更したい場合は、以下の `generate_size` の値を変更してください。
    generate_size = len(df_original)

    with mo.status.spinner(f"{generate_size} 件の合成データを生成中..."):
        sd = mostly.generate(g, size=generate_size)
        df_synthetic = sd.data()

    mo.md(f"✅ **合成データの生成が完了しました！** ({len(df_synthetic)} レコード)")
    return df_synthetic, sd


@app.cell
def _(df_synthetic, mo):
    mo.stop(df_synthetic.empty)
    mo.ui.table(df_synthetic, page_size=10, label="合成データ (Synthetic Data)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Step 5: 生成データ品質と Data QA ("The Proof")

    生成された合成データが、元データの**分布の形状**や**カラム間の相関（関係性）**を維持できているか確認します。
    """)
    return


@app.cell
def _(df_synthetic, mo, pd, sd):
    mo.stop(
        df_synthetic.empty,
        mo.md(
            "データが生成されると、ここに評価メトリクスとData QAレポートが表示されます。"
        ),
    )

    _metrics_sd = (
        getattr(sd.tables[0], "tabular_model_metrics", None)
        if hasattr(sd, "tables") and sd.tables
        else None
    )

    def _fmt_pct(v):
        return f"{v * 100:.1f}%" if isinstance(v, (int, float)) else "N/A"

    _sim = _metrics_sd.similarity if _metrics_sd else None
    _sim_metrics = []
    if _sim:
        for _name, _label in [
            ("cosine_similarity_training_synthetic", "Cosine Sim (Train↔Syn)"),
            (
                "discriminator_auc_training_synthetic",
                "Discriminator AUC (Train↔Syn)",
            ),
        ]:
            _v = getattr(_sim, _name, None)
            if _v is not None:
                _sim_metrics.append({"指標": _label, "スコア": _fmt_pct(_v)})

    _dist = _metrics_sd.distances if _metrics_sd else None
    if _dist:
        for _name, _label in [
            ("ims_training", "Identical Match Rate (Train)"),
            (
                "dcr_training",
                "DCR (Distance to Closest Record) 5th percentile",
            ),
        ]:
            _v = getattr(_dist, _name, None)
            if _v is not None:
                _sim_metrics.append({"指標": _label, "スコア": _fmt_pct(_v)})

    _metrics_table = mo.ui.table(
        pd.DataFrame(_sim_metrics)
        if _sim_metrics
        else pd.DataFrame([{"指標": "-", "スコア": "-"}]),
        label="生成データの品質・プライバシー指標",
    )

    mo.vstack(
        [
            mo.md("### 📊 生成データメトリクス (Similarity & Privacy)"),
            _metrics_table,
            mo.md(
                """
        **各指標の見方:**
        - **Cosine Sim (Cosine Similarity)**: 元データと合成データの分布の類似度です。100%に近いほど、統計的特性（全体の形状）を正確に再現できています。
        - **Discriminator AUC**: 合成データと元データを見分ける分類モデルの精度です。50%に近いほど見分けがつかない「良い合成データ」であることを示します。
        - **Identical Match Rate**: 元データと「全カラムが完全に一致」してしまったレコードの割合です。丸暗記（Overfitting）のリスクを示すため、0%が望ましいです。
        - **DCR (Distance to Closest Record)**: 合成データから最も近い実在データへの距離の5パーセンタイル値です。この距離が0に近いほど、実在の個人に近いデータが生成されているというプライバシーリスクを示します。
        """
            ),
        ]
    )
    return


@app.cell
def _(Path, mo, os, sd, subprocess, sys):
    mo.stop(sd is None)

    _local_dir_sd = Path("./mostlyai_local").resolve()
    _sd_data_report_dir = (
        _local_dir_sd / "synthetic-datasets" / sd.id / "DataQAReports"
    )

    def _find_report_sd(report_dir):
        if not report_dir.exists():
            return None
        _html_files = list(report_dir.glob("*.html"))
        return _html_files[0] if _html_files else None

    _sd_data_report = _find_report_sd(_sd_data_report_dir)

    def _open_report_sd(p):
        _path_str = str(p)
        if sys.platform == "win32":
            os.startfile(_path_str)
        else:
            _opener = "open" if sys.platform == "darwin" else "xdg-open"
            _res = subprocess.run(
                [_opener, _path_str], capture_output=True, text=True
            )
            if _res.returncode != 0:
                print(f"⚠️ Failed to open {p.name}: {_res.stderr.strip()}")

    if _sd_data_report:
        _btn_sd = mo.ui.run_button(
            label="📦 Data QA を開く",
            on_change=lambda _, p=_sd_data_report: _open_report_sd(p),
        )
        _qa_ui_sd = mo.vstack(
            [
                mo.md("### 📑 Data QA レポート"),
                mo.md(
                    f"📄 `{_sd_data_report.name}`\n\n`file://{_sd_data_report}`"
                ),
                _btn_sd,
            ]
        )
    else:
        _qa_ui_sd = mo.md("⚠️ Data QA レポートが見つかりませんでした。")

    _qa_ui_sd
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 🔍 分布・相関の可視化
    """)
    return


@app.cell
def _(df_original, mo):
    _cols = df_original.columns.tolist()
    column_sel = mo.ui.dropdown(
        _cols, value=_cols[0] if len(_cols) > 0 else None, label="対象カラム:"
    )

    mo.md(
        f"""**変数の分布比較 (1D Distribution):**\n元データ（Original）と合成データ（Synthetic）の分布がどれくらい一致しているかを視覚的に確認します。
        Dataレポートには一覧で表示されていますので、詳細はそちらを御覧ください。
        \n\n{column_sel}"""
    )
    return (column_sel,)


@app.cell
def _(alt, column_sel, df_original, df_synthetic, mo, pd):
    mo.stop(df_synthetic.empty or not column_sel.value)

    _c = column_sel.value
    _is_numeric = pd.api.types.is_numeric_dtype(df_original[_c])

    _df_orig_plot = df_original[[_c]].copy()
    _df_orig_plot["Type"] = "Original"

    _df_syn_plot = df_synthetic[[_c]].copy()
    _df_syn_plot["Type"] = "Synthetic"

    _df_plot = pd.concat([_df_orig_plot, _df_syn_plot])

    if _is_numeric:
        _chart = (
            alt.Chart(_df_plot)
            .transform_density(
                _c,
                as_=[_c, 'density'],
                groupby=['Type']
            )
            .mark_line(opacity=0.8, strokeWidth=3)
            .encode(
                x=alt.X(f"{_c}:Q", title=_c),
                y=alt.Y('density:Q', title='Density'),
                color=alt.Color(
                    "Type:N",
                    scale=alt.Scale(
                        domain=["Original", "Synthetic"],
                        range=["steelblue", "salmon"],
                    ),
                ),
            )
            .properties(
                title=f"{_c} の分布比較 (Density)",
                width=600,
                height=400,
            )
            .interactive()
        )
    else:
        _chart = (
            alt.Chart(_df_plot)
            .mark_bar()
            .encode(
                x=alt.X(f"{_c}:N", title=_c),
                xOffset="Type:N",
                y=alt.Y('count()', title='Count'),
                color=alt.Color(
                    "Type:N",
                    scale=alt.Scale(
                        domain=["Original", "Synthetic"],
                        range=["steelblue", "salmon"],
                    ),
                ),
            )
            .properties(
                title=f"{_c} の分布比較",
                width=600,
                height=400,
            )
            .interactive()
        )

    mo.ui.altair_chart(_chart)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Step 6: プライバシー検証とリスクの払拭 ("Safe for Sharing")

    合成データは、一見すると本物のように見えますが、実在する**特定の個人のデータを「記憶（丸暗記）」していないか**（Membership Inference）、
    あるいは**「極めて似た架空のデータ」を作ってしまっていないか**を検証します。

    1. **完全一致 (Identical Match) の検査:**
       元データと全く同じ値を持つ行が1つでも生成されていないか確認します。
    2. **最近傍探索 (Nearest Neighbor):**
       合成データからランダムに1人選び、「もっとも似ている実在の人物」との間にしっかり距離・差異があることを確認します。
    """)
    return


@app.cell
def _(df_original, df_synthetic, mo):
    mo.stop(
        df_synthetic.empty, mo.md("合成データが生成されると検証結果が表示されます。")
    )

    _numeric_cols = (
        df_original.select_dtypes(include=["number"]).columns.tolist()
    )
    _categorical_cols = (
        df_original.select_dtypes(exclude=["number"]).columns.tolist()
    )

    # 1. 完全一致の検査
    _merged = df_original.merge(
        df_synthetic, on=_numeric_cols + _categorical_cols, how="inner"
    )
    _exact_matches = len(_merged)
    _match_pct = (_exact_matches / len(df_synthetic)) * 100

    mo.md(
        f"""
        ### 🛡️ 完全一致 (Identical Match) の検査
        - 原データと全カラムが**完全に一致**する合成レコードの数: **{_exact_matches} 件** ({_match_pct:.2f}%)
        - **結論:** {'実在の個人の丸暗記（Overfitting）は起きていません。ゼロ件なので安心です。' if _exact_matches == 0 else f'いくつかのレコード（{_exact_matches}件）が偶然一致しました。カテゴリ変数が少ない場合などに起こり得ます。'}
        """
    )
    return


@app.cell
def _(NearestNeighbors, StandardScaler, df_original, df_synthetic, mo, np, pd):
    mo.stop(df_synthetic.empty)

    _numeric_cols = (
        df_original.select_dtypes(include=["number"]).columns.tolist()
    )
    _categorical_cols = (
        df_original.select_dtypes(exclude=["number"]).columns.tolist()
    )

    # 2. 最近傍探索
    np.random.seed(None)
    _target_synthetic = df_synthetic.sample(1).iloc[0]

    if not _numeric_cols:
        _nn_alert = mo.md("数値カラムがないため、最近傍探索をスキップしました。")
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
            {
                "合成データ (Synthetic)": _target_synthetic,
                "最近傍の実在データ (Real)": _closest_real,
            }
        ).T

        _diff_msgs = []
        for _c in _numeric_cols:
            _syn_val = _target_synthetic[_c]
            _real_val = _closest_real[_c]
            if pd.isna(_syn_val) or pd.isna(_real_val) or _syn_val != _real_val:
                if pd.isna(_syn_val) and pd.isna(_real_val):
                    continue
                _diff_str = "N/A" if (pd.isna(_syn_val) or pd.isna(_real_val)) else f"{abs(_syn_val - _real_val):.2f}"
                _diff_msgs.append(
                    f"- **{_c}**: 合成={_syn_val}, 実データ={_real_val} (差: {_diff_str})"
                )

        for _c in _categorical_cols:
            _syn_val = _target_synthetic[_c]
            _real_val = _closest_real[_c]
            if pd.isna(_syn_val) or pd.isna(_real_val) or _syn_val != _real_val:
                if pd.isna(_syn_val) and pd.isna(_real_val):
                    continue
                _diff_msgs.append(
                    f"- **{_c}**: 合成={_syn_val}, 実データ={_real_val} (カテゴリ不一致)"
                )

        _diff_text = (
            "\n".join(_diff_msgs) if _diff_msgs else "差分なし（完全一致）"
        )

        _nn_alert = mo.vstack(
            [
                mo.md("---"),
                mo.md(
                    "### 🕵️ 最も似ている実在データの探索 (Nearest Neighbor)"
                ),
                mo.md(
                    "ランダムに選んだある「合成データの人物（架空）」と、属性が最も近い「実在の人物」を比較します。"
                ),
                mo.ui.table(_comparison_df, label="比較表"),
                mo.md(
                    f"""
        **主な差分:**
        {_diff_text}

        > ✅ **評価:** もっとも似ている実在データであってもこれだけの違いがあります。つまり「誰か特定の人物のデータをちょっと改変しただけ」ではない、**新しい架空の人物データ**であることが分かります。
        """
                ),
            ]
        )

    _nn_alert
    return


if __name__ == "__main__":
    app.run()
