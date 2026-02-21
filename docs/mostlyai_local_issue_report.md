# MostlyAI Local: PULL_TRAINING_DATA エラーの調査報告と解決策

## 概要
MostlyAI SDK (Local Mode) を使用して、UCI Bank Marketing データセットから合成データを生成する際、学習プロセスが `PULL_TRAINING_DATA` のステップで `0%` のまま停止（内部的にはクラッシュ）する事象が発生しました。

本ドキュメントでは、このエラーの根本原因、調査プロセス、および解決策についてまとめます。

## 事象の再現条件
1. `mostlyai[local]` (バージョン 5.10.1) を使用。
2. 対象データセット: `bank-marketing.csv`
3. 実行コード: `mostly.train(data=df)`

上記を実行すると、コンソール上では以下のようなログを表示したままプロセスが進まなくなるか、Pythonプロセスごと終了する挙動を示しました。

```
Started generator training
Overall job progress                               ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━        0%  -:--:--
Step data:tabular PULL_TRAINING_DATA               ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━        0%  -:--:--
```

## エラーの根本原因 (Root Cause)

**Pandas の `OutOfBoundsDatetime` 例外によるクラッシュ** でした。

対象データセットの `month` カラムには `"may"`, `"jun"` といった「月」を表す文字列が格納されています。
MostlyAI はデータを読み込む際、各カラムの型を自動推論します。このとき、`month` カラムを誤って **`TABULAR_DATETIME` (日時型)** として認識してしまいました。

内部処理では、Pandas の `to_datetime` を使用してこのカラムをパースしようと試みます。
しかし、年（Year）の情報がないため、Pandas はデフォルトでこれを **0001年**（例: `0001-05-01`）と解釈します。

Pandas のタイムスタンプ（ナノ秒精度）において表現できる最古の日付は **1677年9月21日** です。そのため、0001年という日付は `OutOfBoundsDatetime` となり、例外が発生して処理が強制終了していました。

```python
pandas.errors.OutOfBoundsDatetime: Out of bounds nanosecond timestamp: 1-04-01 00:00:00
```

## 調査プロセス
通常であれば、このようなエラーはトレースバックとして標準出力に表示されるべきですが、MostlyAI Local のバックグラウンド実行（`cli.py` またはサブプロセス経由）の仕組みと例外ハンドリングの仕様により、ターミナルの画面上にはエラーが握り潰されて露出しない状態になっていました。

そのため、以下の手順で強引にエラーを可視化して原因を特定しました。

1. **モンキーパッチの試みとローカルサーバーの把握:**
   最初、SDKのメソッド（`execute_training_job`）をモンキーパッチして標準エラーを乗っ取ろうとしましたが、これは別スレッド・別プロセスで実行されているため効果がありませんでした。
2. **CLIの直接実行:**
   FastAPI のルーティングコード（`routes.py`）を解析した結果、実際の学習処理は `cli.py run-training <generator_id> <home_dir>` としてサブプロセスからキックされていることが判明しました。
3. **SDKへのファイル出力パッチ:**
   `mostlyai/sdk/_local/execution/jobs.py` の `execute_training_job` 関数内の `except Exception as e:` ブロックに直接手を加え、エラー時に `traceback.format_exc()` の結果をファイル（`real_error.txt`）へと強制的に書き出すようにSDKのソースコードを緊急修正しました。
4. **トレースログの取得:**
   再度学習を実行し、出力された `real_error.txt` を確認した結果、前述の `pandas.errors.OutOfBoundsDatetime` が発生していることが明確になりました。

## 解決策 (Solution)

エラーを防ぐためには、`month` カラムが `TABULAR_DATETIME` ではなく `TABULAR_CATEGORICAL` (カテゴリ型) として扱われるよう明示的に指定する必要があります。

SDKに対して、辞書型の設定（Config）を `train()` メソッドに渡すことで、特定のカラムのエンコーディングタイプを上書き指定することが可能です。

## 備考
- `jobs.log` などの内部ログに直接アクセスできない環境やアーキテクチャでは、こうしたライブラリのバグやエッジケースに遭遇した際に「どこでプロセスが死んだか」を追うのが非常に困難になります。
- 今回採用した「ライブラリのソースコードに直接書き出し処理を埋め込む」というアプローチは、ブラックボックス化された OSS ツールをデバッグする上で強力な手段となります。
