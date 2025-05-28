import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pickle
import time
import great_expectations as gx


# E302修正: classの前に2行の空白
class DataLoader:
    """データロードを行うクラス"""

    @staticmethod
    def load_titanic_data(path=None):
        """Titanicデータセットを読み込む"""
        if path:
            return pd.read_csv(path)
        else:
            relative_script_path = "data/Titanic.csv"
            if os.path.exists(relative_script_path):
                return pd.read_csv(relative_script_path)
            else:
                # E501修正: 長い行を改行
                print(
                    f"警告: 指定された相対パスにファイルが見つかりません: "
                    f"{relative_script_path} (現在の作業ディレクトリ: "
                    f"{os.getcwd()})"
                )
                return None

    @staticmethod
    def preprocess_titanic_data(data):
        """Titanicデータを前処理する"""
        data = data.copy()
        columns_to_drop = []
        for col in ["PassengerId", "Name", "Ticket", "Cabin"]:
            if col in data.columns:
                columns_to_drop.append(col)
        if columns_to_drop:
            data.drop(columns_to_drop, axis=1, inplace=True)
        if "Survived" in data.columns:
            y = data["Survived"]
            X = data.drop("Survived", axis=1)
            return X, y
        else:
            return data, None


class DataValidator:
    """データバリデーションを行うクラス"""

    @staticmethod
    def validate_titanic_data(data):
        """Titanicデータセットの検証"""
        if not isinstance(data, pd.DataFrame):
            return False, ["データはpd.DataFrameである必要があります"]
        try:
            context = gx.get_context()
            data_source = context.data_sources.add_pandas("pandas")
            data_asset = data_source.add_dataframe_asset(
                name="pd dataframe asset"
            )
            batch_definition = data_asset.add_batch_definition_whole_dataframe(
                "batch definition"
            )
            batch = batch_definition.get_batch(
                batch_parameters={"dataframe": data}
            )
            results = []
            required_columns = [
                "Pclass", "Sex", "Age", "SibSp",
                "Parch", "Fare", "Embarked",
            ]
            missing_columns = [
                col for col in required_columns if col not in data.columns
            ]
            if missing_columns:
                print(
                    f"警告: 以下のカラムがありません: "
                    f"{missing_columns}"
                )
                return False, [{"success": False,
                                "missing_columns": missing_columns}]
            expectations = [
                gx.expectations.ExpectColumnDistinctValuesToBeInSet(
                    column="Pclass", value_set=[1, 2, 3]
                ),
                gx.expectations.ExpectColumnDistinctValuesToBeInSet(
                    column="Sex", value_set=["male", "female"]
                ),
                gx.expectations.ExpectColumnValuesToBeBetween(
                    column="Age", min_value=0, max_value=100
                ),
                gx.expectations.ExpectColumnValuesToBeBetween(
                    column="Fare", min_value=0, max_value=600
                ),
                gx.expectations.ExpectColumnDistinctValuesToBeInSet(
                    column="Embarked", value_set=["C", "Q", "S", ""]
                ),
            ]
            for expectation in expectations:
                result = batch.validate(expectation)
                results.append(result)
            is_successful = all(result.success for result in results)
            return is_successful, results
        except Exception as e:
            error_message = f"Great Expectations検証エラー: {e}"
            print(error_message[:75] + "..."
                  if len(error_message) > 78 else error_message)
            return False, [{"success": False, "error": str(e)}]


class ModelTester:
    """モデルテストを行うクラス"""

    @staticmethod
    def create_preprocessing_pipeline():
        """前処理パイプラインを作成"""
        numeric_features = ["Age", "Fare", "SibSp", "Parch"]
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_features = ["Pclass", "Sex", "Embarked"]
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ],
            remainder="drop",  # 指定されていない列は削除
        )
        return preprocessor

    @staticmethod
    def train_model(X_train, y_train, model_params=None):
        """モデルを学習する"""
        if model_params is None:
            model_params = {"n_estimators": 100, "random_state": 42}
        preprocessor = ModelTester.create_preprocessing_pipeline()
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(**model_params)),
            ]
        )
        model.fit(X_train, y_train)
        return model

    @staticmethod
    def evaluate_model(model, X_test, y_test):
        """モデルを評価する"""
        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - start_time
        accuracy = accuracy_score(y_test, y_pred)
        return {"accuracy": accuracy, "inference_time": inference_time}

    @staticmethod
    def save_model(model, path="models/titanic_model.pkl"):
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "titanic_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        return path

    @staticmethod
    def load_model(path="models/titanic_model.pkl"):
        """モデルを読み込む"""
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model

    @staticmethod
    def compare_with_baseline(current_metrics, baseline_threshold=0.75):
        """ベースラインと比較する"""
        return current_metrics["accuracy"] >= baseline_threshold


def test_data_validation():
    """データバリデーションのテスト"""
    data = DataLoader.load_titanic_data()
    X, y = DataLoader.preprocess_titanic_data(data)
    success, results = DataValidator.validate_titanic_data(X)
    assert success, "データバリデーションに失敗しました"
    bad_data = X.copy()
    bad_data.loc[0, "Pclass"] = 5  # 明らかに範囲外の値
    success, results = DataValidator.validate_titanic_data(bad_data)
    assert not success, "異常データをチェックできませんでした"


def test_model_performance():
    """モデル性能のテスト"""
    data = DataLoader.load_titanic_data()
    X, y = DataLoader.preprocess_titanic_data(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = ModelTester.train_model(X_train, y_train)
    metrics = ModelTester.evaluate_model(model, X_test, y_test)
    error_msg_baseline = (
        f"モデル性能がベースラインを下回っています: {metrics['accuracy']}"
    )
    assert ModelTester.compare_with_baseline(
        metrics, 0.75
    ), error_msg_baseline
    error_msg_time = (
        f"推論時間が長すぎます: {metrics['inference_time']}秒"
    )
    assert metrics["inference_time"] < 1.0, error_msg_time


def test_inference_speed_and_accuracy():
    """推論時間と精度をチェックする新しいテスト関数"""
    print("\n実行中: test_inference_speed_and_accuracy")
    data = DataLoader.load_titanic_data()
    if data is None:
        assert (
            False
        ), ("テスト用データのロードに失敗しました "
            "(test_inference_speed_and_accuracy)")
    X, y = DataLoader.preprocess_titanic_data(data)
    if X is None or y is None:
        assert (
            False
        ), ("テスト用データの前処理に失敗しました "
            "(test_inference_speed_and_accuracy)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model_params = {"n_estimators": 100, "random_state": 42}
    model = ModelTester.train_model(
        X_train, y_train, model_params=model_params
    )
    if model is None:
        assert False, ("モデルの学習に失敗しました "
                       "(test_inference_speed_and_accuracy)")
    metrics = ModelTester.evaluate_model(model, X_test, y_test)
    accuracy = metrics["accuracy"]
    inference_time = metrics["inference_time"]
    print(f"テスト結果 - 精度: {accuracy:.4f}")
    print(f"テスト結果 - 推論時間: {inference_time:.4f}秒")
    error_msg_acc = (
        f"精度 ({accuracy:.4f}) が閾値 (0.70) を下回っています。"
    )
    assert accuracy > 0.70, error_msg_acc
    error_msg_time = (
        f"推論時間 ({inference_time:.4f}秒) "
        f"が閾値 (1.0秒) を超えています。"
    )
    assert inference_time < 1.0, error_msg_time


if __name__ == "__main__":
    data = DataLoader.load_titanic_data()
    X, y = DataLoader.preprocess_titanic_data(data)
    success, results = DataValidator.validate_titanic_data(X)
    # E501修正: 277行目あたり
    status_message = '成功' if success else '失敗'
    print(f"データ検証結果: {status_message}")
    for result in results:
        if not result["success"]:
            error_type = result.get('expectation_config', {}).get('type', 'N/A')
            error_details = str(result)[:50] + "..."
            print(f"異常タイプ: {error_type}, 結果: {error_details}")
    if not success:
        print("データ検証に失敗しました。処理を終了します。")
        exit(1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model_params = {"n_estimators": 100, "random_state": 42}
    model = ModelTester.train_model(X_train, y_train, model_params)
    metrics = ModelTester.evaluate_model(model, X_test, y_test)
    print(f"精度: {metrics['accuracy']:.4f}")
    print(f"推論時間: {metrics['inference_time']:.4f}秒")
    model_path = ModelTester.save_model(model)
    baseline_ok = ModelTester.compare_with_baseline(metrics)
    print(f"ベースライン比較: {'合格' if baseline_ok else '不合格'}")

    

# W292修正: ファイルの最後に空行が1行あることを確認 (この行が最後の行なので、この下にエディタで空行を追加)