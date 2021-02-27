import dash_core_components as dcc
import dash_html_components as html
from utils import Header, make_dash_table
import pandas as pd
import pathlib
import plotly.graph_objs as go

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../data").resolve()

df_dividend = pd.read_csv(DATA_PATH.joinpath("df_dividend.csv"))
df_realized = pd.read_csv(DATA_PATH.joinpath("df_realized.csv"))
df_unrealized = pd.read_csv(DATA_PATH.joinpath("df_unrealized.csv"))

df_graph=pd.read_csv("/home/grover/Documents/pythondoc/dashboards/FOREX1-DASH/dataX/CAD=X.csv")
df_graph2=pd.read_csv("/home/grover/Documents/pythondoc/dashboards/FOREX1-DASH/dataX/JPY=X.csv")
#df_graph.value[0:1000]
def create_layout(app):
    return html.Div(
        [
            Header(app),
            # page 5
            html.Div(
                [
                    # Row 1
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        ["Correlation graph"], className="subtitle padded"
                                    ),
                                    html.P(
                                        [
                                            "This work presents the correlation between the two currencies, it is possible to observe that the correlation is non-linear, therefore for a predictive system it is necessary to take the behavior and dependence of these two variables"
                                        ],
                                        style={"color": "#7a7a7a"},
                                    ),
                                ],
                                className="twelve columns",
                            )
                        ],
                        className="row ",
                    ),
                    # Row 2
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Br([]),
                                    html.H6(
                                        ["Correlation between ttw variables in the price"],
                                        className="subtitle tiny-header padded",
                                    ),
                                    html.Div(
                                        [
                                            dcc.Graph(
                                        id="graph-4",
                                        figure={
                                            "data": [
                                                go.Scatter(
                                                    x=df_graph.Open[0:1000],
                                                    y=df_graph2.Open[0:1000],
                                                    line={"color": "#97151c"},
                                                    mode="markers",
                                                    name="USD-CAD vs USDJPY",
                                                ),
                                         
                                            ],
                                            "layout": go.Layout(
                                                autosize=True,
                                                width=700,
                                                height=200,
                                                font={"family": "Raleway", "size": 10},
                                                margin={
                                                    "r": 30,
                                                    "t": 30,
                                                    "b": 30,
                                                    "l": 30,
                                                },
                                                showlegend=True,
                                                titlefont={
                                                    "family": "Raleway",
                                                    "size": 10,
                                                },
                                             
                                            ),
                                        },
                                        config={"displayModeBar": False},
                                    ),
                                        ],
                                        style={"overflow-x": "auto"},
                                    ),
                                ],
                                className="twelve columns",
                            )
                        ],
                        className="row ",
                    ),
                    # Row 3
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        ["Realized/unrealized gains as of 01/31/2018"],
                                        className="subtitle tiny-header padded",
                                    )
                                ],
                                className=" twelve columns",
                            )
                        ],
                        className="row ",
                    ),
                    # Row 4
                    html.Div(
                        [
                            html.Div(
                                [html.Table(make_dash_table(df_realized))],
                                className="six columns",
                            ),
                            html.Div(
                                [html.Table(make_dash_table(df_unrealized))],
                                className="six columns",
                            ),
                        ],
                        className="row ",
                    ),
                ],
                className="sub_page",
            ),
        ],
        className="page",
    )
