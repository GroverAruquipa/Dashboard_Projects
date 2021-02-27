import dash_html_components as html
from utils import Header, make_dash_table
import pandas as pd
import pathlib
import dash
import dash_table
# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../data").resolve()

df_dividend = pd.read_csv(DATA_PATH.joinpath("df_dividend.csv"))
df_realized = pd.read_csv(DATA_PATH.joinpath("df_realized.csv"))
df_unrealized = pd.read_csv(DATA_PATH.joinpath("df_unrealized.csv"))
df_graph = pd.read_csv(DATA_PATH.joinpath("df_graph.csv")) ##CHANGE
df_graph=pd.read_csv("/home/grover/Documents/pythondoc/dashboards/FOREX1-DASH/dataX/CAD=X.csv")
df_graph2=pd.read_csv("/home/grover/Documents/pythondoc/dashboards/FOREX1-DASH/dataX/JPY=X.csv")

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
                                        ["DATASE CAD/USD"], className="subtitle padded"
                                    ),
                                    html.P(
                                        [
                                            "Datasate available in Yahoo FInance"
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
                                        ["Date,value, Open"],
                                        className="subtitle tiny-header padded",
                                    ),
                                    html.Div(
                                        [  
                                            html.Table(
                                            make_dash_table(df_graph),
                                            className="tiny-header",
                                            )

                                        ],
                                        style={"overflow-x": "auto",
                                                'height': '350px',
                                                 'overflow': 'scroll',},
                                    ),
                                ],
                                className="twelve columns",
                            )
                        ],
                        className="row ",
                    ),
                    # Row 3
                    html.Div([
                        html.H6(["Data saved"])



                    ])
           
                ],
                className="sub_page",
            ),
        ],
        className="page",
    )