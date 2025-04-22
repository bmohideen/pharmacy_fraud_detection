import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import joblib
import plotly.express as px
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

# Loading data and models
df_raw = joblib.load("raw_df.pkl")
X_test, y_test = joblib.load("test_split.pkl")
model_dict = {
    "Pruned Decision Tree": joblib.load("model_dt_pruned.pkl"),
    "Random Forest":         joblib.load("model_rf.pkl"),
    "AdaBoost":              joblib.load("model_ab.pkl")
}

# Ascending claims for drop-down menu
sorted_claims = sorted(X_test.index)

model_accuracy = {
    "Pruned Decision Tree": 0.9994,
    "Random Forest":        0.9994,
    "AdaBoost":             0.9994
}

province_encoder = {'AB':0,'ON':1,'MB':2,'BC':3,'QC':4}
therapyclass_encoder = {
    'Antidiabetics':0,'Antibiotics':1,'Antihypertensives':2,
    'Cholesterol Reducers':3,'Antiinflammatories':4,'Respiratory':5,
    'Analgesics':6,'Gastrointestinal':7,'Hormonal':8,'Antidepressants':9
}

# Initializing the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.YETI],
    suppress_callback_exceptions=True
)
server = app.server

# Dash app layout
app.layout = html.Div(
    style={"backgroundColor": "#d0e7f9", "minHeight": "100vh", "padding": "0"},
    children=[
        dbc.Container(
            fluid=True,
            style={"maxWidth": "1200px", "margin": "auto", "padding": "20px"},
            children=[
                html.H2(
                    "Pharmacy Fraud Detection",
                    style={"textAlign": "center", "color": "#2C3E50", "marginBottom": "20px", "fontWeight": "900"}
                ),

                dcc.Tabs(
                    id="tabs", value="tab-1",
                    children=[
                        dcc.Tab(
                            label="Predict a Claim", value="tab-1",
                            style={"padding": "10px"},
                            selected_style={"backgroundColor": "#008CBA", "color": "white"}
                        ),
                        dcc.Tab(
                            label="Model Performance", value="tab-2",
                            style={"padding": "10px"},
                            selected_style={"backgroundColor": "#008CBA", "color": "white"}
                        ),
                        dcc.Tab(
                            label="EDA by Fraud Status", value="tab-3",
                            style={"padding": "10px"},
                            selected_style={"backgroundColor": "#008CBA", "color": "white"}
                        ),
                    ],
                    style={"marginBottom": "20px"}
                ),

                html.Div(id="tabs-content")
            ]
        )
    ]
)

# Rendering the tab contents
@app.callback(Output("tabs-content", "children"),
              Input("tabs", "value"))
def render_tab_content(tab):

    # Tab 1: Predict a Claim
    if tab == "tab-1":
        options = [{"label": str(i+1), "value": cid}
                   for i, cid in enumerate(sorted_claims)]

        return dbc.Card(
            style={"backgroundColor": "#f0f8ff"},
            className="shadow-sm p-4 mb-4 rounded",
            children=[
                dbc.CardBody([
                    html.H4("Select Model", style={"textAlign": "center", "color": "#34495E"}),
                    dcc.Dropdown(
                        id="model-select",
                        options=[{"label": k, "value": k} for k in model_dict],
                        value="Pruned Decision Tree",
                        className="mb-3"
                    ),

                    html.H5(
                        "Select Existing Claim or Enter a Custom Claim:",
                        style={"textAlign": "center", "marginTop": "20px"}
                    ),
                    dcc.Dropdown(
                        id="existing-claim-dropdown",
                        options=options,
                        placeholder="Pick a claim number…",
                        className="mb-3"
                    ),

                    html.Div("Or enter manually:", style={"textAlign": "center", "marginBottom": "10px"}),
                    dbc.Row([
                        dbc.Col([dbc.Label("DIN_Submitted"), dcc.Input(id="input-DIN_Submitted", type="number", value=0)]),
                        dbc.Col([dbc.Label("Qty"),           dcc.Input(id="input-Qty",           type="number", value=1)]),
                        dbc.Col([dbc.Label("TotalPaid"),     dcc.Input(id="input-TotalPaid",     type="number", value=0)]),
                        dbc.Col([dbc.Label("DIN_Paid"),      dcc.Input(id="input-DIN_Paid",      type="number", value=0)]),
                        dbc.Col([dbc.Label("Province"),      dcc.Dropdown(id="input-Province", options=[{"label":p,"value":p} for p in province_encoder], value="ON")]),
                        dbc.Col([dbc.Label("TherapyClass"),  dcc.Dropdown(id="input-TherapyClass", options=[{"label":t,"value":t} for t in therapyclass_encoder], value="Antibiotics")]),
                    ], className="g-3 mb-3"),
                    html.Div(
                        dbc.Button("PREDICT", id="predict-button", color="primary", size="lg"),
                        style={"textAlign": "center", "marginTop": "20px", "marginBottom": "20px"}
                    ),

                    html.Div(id="prediction-output", style={"fontSize": 24, "textAlign": "center"}),
                    html.Div(id="accuracy-display",  style={"fontSize": 18, "color": "gray", "textAlign": "center"})
                ])
            ]
        )

    # Tab 2: Model Performance
    elif tab == "tab-2":
        return dbc.Card(
            style={"backgroundColor": "#f0f8ff"},
            className="shadow-sm p-4 mb-4 rounded",
            children=[
                dbc.CardBody([
                    html.H4("Model Performance Comparison", style={"textAlign": "center", "color": "#34495E"}),
                    dcc.Dropdown(
                        id="model-metrics-select",
                        options=[{"label": k, "value": k} for k in model_dict],
                        value="Pruned Decision Tree",
                        className="mb-4"
                    ),
                    dbc.Row([dbc.Col(dcc.Graph(id="confusion-matrix-plot")), dbc.Col(dcc.Graph(id="roc-curve-plot"))]),
                    html.Div(id="metrics-output", style={"fontSize": 16, "marginTop": "20px", "textAlign": "center"})
                ])
            ]
        )

    # Tab 3: EDA by Fraud Status
    else:
        return dbc.Card(
            style={"backgroundColor": "#f0f8ff"},
            className="shadow-sm p-4 mb-4 rounded",
            children=[
                dbc.CardBody([
                    html.H4("Exploratory Data Analysis by Fraud Status", style={"textAlign": "center", "color": "#34495E"}),
                    html.P("Select a variable to visualize by fraud status:", style={"textAlign": "center"}),
                    dcc.Dropdown(
                        id="eda-variable-select",
                        options=[{"label": c, "value": c}
                                 for c in df_raw.columns
                                 if c not in ["ClaimID", "ClaimantID", "IsFraud", "DIN"]],
                        placeholder="Choose a variable…",
                        className="mb-4"
                    ),
                    dcc.Graph(id="eda-plot")
                ])
            ]
        )

# Callback for Tab 1
@app.callback(
    [Output("prediction-output", "children"), Output("accuracy-display", "children")],
    Input("predict-button", "n_clicks"),
    State("model-select", "value"),
    State("existing-claim-dropdown", "value"),
    State("input-DIN_Submitted", "value"),
    State("input-Qty", "value"),
    State("input-TotalPaid", "value"),
    State("input-DIN_Paid", "value"),
    State("input-Province", "value"),
    State("input-TherapyClass", "value")
)
def predict_fraud(nc, mdl, cid, din_s, qty, tot, din_p, prov, ther):
    if not nc:
        return "", ""
    if cid is not None:
        r = X_test.loc[cid]
        pu, tp, dp, pr, th = r["pack_unit"], r["TotalPaid"], r["DIN_Paid"], r["Province"], r["TherapyClass"]
    else:
        pu = din_s/qty if qty else 0
        tp, dp, pr, th = tot or 0, din_p or 0, prov, ther
    pe = province_encoder.get(pr, -1)
    te = therapyclass_encoder.get(th, -1)
    arr = np.array([[pu, tp, dp, pe, te]])
    p = model_dict[mdl].predict(arr)[0]
    label = "FRAUDULENT" if p == 1 else "NOT FRAUDULENT"
    acc = model_accuracy[mdl] * 100
    return label, f"NOTE: The accuracy for the selected model is {acc:.2f}%"

# Callback for Tab 2
@app.callback(
    [Output("confusion-matrix-plot", "figure"),
     Output("roc-curve-plot", "figure"),
     Output("metrics-output", "children")],
    Input("model-metrics-select", "value")
)
def update_model_performance(mdl):
    m = model_dict[mdl]
    yp = m.predict(X_test)
    pr = m.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, yp)
    cm_fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                       labels={"x": "Predicted", "y": "Actual"})
    cm_fig.update_layout(title=f"Confusion Matrix — {mdl}", title_x=0.5)
    fpr, tpr, _ = roc_curve(y_test, pr)
    roc_auc = auc(fpr, tpr)
    roc_fig = px.area(x=fpr, y=tpr,
                      title=f"ROC Curve — {mdl} (AUC = {roc_auc:.2f})",
                      labels={"x": "False Positive Rate", "y": "True Positive Rate"},
                      template="plotly_white")
    roc_fig.update_layout(title_x=0.5)
    rpt = classification_report(y_test, yp, output_dict=True)
    txt = (f"Accuracy: {rpt['accuracy']:.2f} | "
           f"Precision (Fraud): {rpt['1']['precision']:.2f} | "
           f"Recall (Fraud): {rpt['1']['recall']:.2f} | "
           f"F1-score (Fraud): {rpt['1']['f1-score']:.2f}")
    return cm_fig, roc_fig, txt

# Callback for Tab 3
@app.callback(
    Output("eda-plot", "figure"),
    Input("eda-variable-select", "value")
)
def update_eda_plot(var):
    if not var:
        return {}
    df = df_raw.loc[X_test.index].copy()
    df["IsFraud"] = y_test.values
    cats = ["PharmacyChainName","PharmacyName","City","Province","TherapyClass","Postcode"]
    if var in cats:
        fig = px.histogram(df, x=var, color="IsFraud", barmode="group",
                           title=f"{var} Distribution by Fraud Status",
                           labels={"IsFraud":"Fraud", var:var},
                           template="plotly_white")
    else:
        fig = px.box(df, x="IsFraud", y=var, color="IsFraud",
                     title=f"{var} by Fraud Status",
                     labels={"IsFraud":"Fraud", var:var},
                     template="plotly_white")
    fig.update_layout(title_x=0.5, transition_duration=300)
    return fig

if __name__ == "__main__":
    app.run_server(debug=True)

