"""Browser-based interface for the existing Wine Quality Prediction System.
The original CLI and its ten modules remain unchanged; this file adds a web UI
that calls the same data preparation, Random Forest, predictor and statistics code.
"""
import os, sys, random
from flask import Flask, render_template, request, redirect, url_for, flash

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import config, data_loader, preprocessing, predictor, statistics_module
from src import model as model_module

app = Flask(__name__)
app.secret_key = os.environ.get("WINE_APP_SECRET", "wine-quality-local-demo-key")

# Train once when the server starts, matching the original CLI bootstrap flow.
DATASET_PATH = data_loader.find_dataset_path()
prepared = preprocessing.prepare_dataset(DATASET_PATH)
model = model_module.train_model(prepared)
evaluation = model_module.evaluate_model(model, prepared)
model_module.save_bundle(model, prepared, evaluation)


def prediction_for_values(values):
    label, confidence, probabilities = predictor.predict_features(
        model, prepared.scaler, values, prepared.label_encoder
    )
    return {"label": str(label), "confidence": float(confidence),
            "probabilities": sorted(
                [{"class": str(k), "probability": float(v)} for k, v in probabilities.items()],
                key=lambda item: item["probability"], reverse=True),
            "features": list(zip(prepared.feature_columns, values))}


def row_result(index):
    values, actual = predictor.get_row_by_index(
        prepared.df, prepared.feature_columns, prepared.target_column, index
    )
    result = prediction_for_values(values)
    result.update({"index": int(index), "actual": str(actual),
                   "matches": str(actual) == str(result["label"])})
    return result


@app.route('/')
def index():
    info = statistics_module.basic_info(prepared.df, prepared.target_column)
    counts = statistics_module.class_balance(prepared.df, prepared.target_column)
    return render_template('index.html', title=config.APP_TITLE, info=info,
                           features=prepared.feature_columns,
                           classes=prepared.class_names,
                           accuracy=evaluation.accuracy,
                           class_counts=[{"label": str(k), "count": int(v)} for k, v in counts.items()])


@app.route('/module/<int:module_id>', methods=['GET', 'POST'])
def module(module_id):
    if module_id < 1 or module_id > 9:
        return redirect(url_for('index'))
    title_map = {
        1: 'Enter Wine Values Manually', 2: 'Select a Sample from Dataset',
        3: 'Random Dataset Sample', 4: 'View Wine Quality Distribution',
        5: 'Test Model with Dataset Samples', 6: 'Model Performance',
        7: 'Feature Importance', 8: 'Dataset Statistics', 9: 'Search Dataset by Quality'
    }
    ctx = {"module_id": module_id, "module_title": title_map[module_id],
           "row_count": len(prepared.df), "features": prepared.feature_columns, "classes": prepared.class_names,
           "result": None, "error": None, "table": None}
    df = prepared.df

    if request.method == 'POST':
        try:
            if module_id == 1:
                values = []
                for feature in prepared.feature_columns:
                    raw = request.form.get(feature, '').strip()
                    if not raw:
                        raise ValueError(f"Please enter a value for {feature}.")
                    values.append(float(raw))
                ctx['result'] = prediction_for_values(values)
            elif module_id == 2:
                index = int(request.form.get('index', '0'))
                if index < 0 or index >= len(df):
                    raise ValueError(f"Choose a row index from 0 to {len(df)-1}.")
                ctx['result'] = row_result(index)
            elif module_id == 3:
                ctx['result'] = row_result(random.randrange(len(df)))
            elif module_id == 5:
                raw_n = request.form.get('n', '10').strip()
                n = int(raw_n or '10')
                if n < 1 or n > 200:
                    raise ValueError('Choose between 1 and 200 samples.')
                results, accuracy_pct = predictor.batch_test_random_samples(
                    model, prepared.scaler, df, prepared.feature_columns,
                    prepared.target_column, prepared.label_encoder, n)
                ctx['table'] = results
                ctx['batch_accuracy'] = accuracy_pct
            elif module_id == 9:
                value = request.form.get('quality', '').strip()
                if not value:
                    raise ValueError('Enter a quality value to search for.')
                matches = data_loader.load_raw_dataframe(DATASET_PATH)
                matches.columns = [str(c).strip() for c in matches.columns]
                matches = matches[matches[prepared.target_column].astype(str).str.strip() == value]
                ctx['search_value'] = value
                ctx['table_columns'] = list(matches.columns)
                ctx['table_rows'] = matches.head(100).to_dict(orient='records')
                ctx['match_count'] = len(matches)
        except (ValueError, TypeError) as exc:
            ctx['error'] = str(exc)
        except Exception as exc:
            ctx['error'] = f"Could not complete this action: {exc}"

    if module_id == 4:
        counts = statistics_module.class_balance(df, prepared.target_column)
        ctx['counts'] = [{"label": str(k), "count": int(v)} for k, v in counts.items()]
        ctx['chart_labels'] = [str(k) for k in counts.index]
        ctx['chart_values'] = [int(v) for v in counts.values]
    elif module_id == 6:
        ev = evaluation
        ctx['metrics'] = [
            ('Accuracy', ev.accuracy), ('Precision (weighted)', ev.precision),
            ('Recall (weighted)', ev.recall), ('F1 Score (weighted)', ev.f1)]
        ctx['report'] = ev.report
        ctx['confusion'] = ev.confusion.tolist()
        ctx['class_names'] = ev.class_names
    elif module_id == 7:
        ctx['importance'] = [{"feature": name, "value": float(value)}
            for name, value in model_module.get_feature_importance(model, prepared.feature_columns)]
    elif module_id == 8:
        ctx['info'] = statistics_module.basic_info(df, prepared.target_column)
        ctx['summary'] = statistics_module.numeric_summary(df, prepared.feature_columns).round(4).reset_index().rename(columns={'index':'feature'}).to_dict(orient='records')
        ctx['correlations'] = [{"feature": str(k), "value": float(v)} for k, v in statistics_module.correlation_with_target(df, prepared.feature_columns, prepared.target_column).items()]
    return render_template('module.html', **ctx)


@app.route('/health')
def health():
    return {"status": "ok", "dataset_rows": len(prepared.df), "model": "RandomForestClassifier"}


if __name__ == '__main__':
    print('\nWine Quality Prediction System website is starting...')
    print(f'Dataset: {DATASET_PATH}')
    print('Open http://127.0.0.1:5000 in your browser.')
    app.run(host='127.0.0.1', port=int(os.environ.get('PORT', '5000')), debug=False)
