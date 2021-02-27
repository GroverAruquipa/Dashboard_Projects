import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from pages import (
    overview,
    graph_analysis,
    correlation,
    indicators,
    documentation_tables,
)
app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)
server=app.server

app.layout=html.Div(
    [dcc.Location(id="url", refresh=False), html.Div(id="page-content")]

)

@app.callback(Output("page-content","children"),[Input("url","pathname")])
def display_page(pathname):
   
    if pathname == "/FOREX1-DASH/correlation":
        return correlation.create_layout(app)
    elif pathname == "/FOREX1-DASH/documentation_tables":
        return documentation_tables.create_layout(app)
    elif pathname == "/FOREX1-DASH/graph_analysis":
        return graph_analysis.create_layout(app)
    elif pathname == "/FOREX1-DASH/indicators":
        return indicators.create_layout(app)
    elif pathname == "/FOREX1-DASH/full-view":
        return(
            overview.create_layout(app),
            graph_analysis.create_layout(app),
            correlation.create_layout(app),
            indicators.create_layout(app),
            anomaly_detection.create_layout(app),
            documentation_tables.create_layout(app),
        )
    else:
        return overview.create_layout(app)

if __name__ == "__main__":
    app.run_server(debug=True)









