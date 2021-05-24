import dash_html_components as html
import dash_core_components as dcc


def Header(app):
    return html.Div([get_header(app), html.Br([]), get_menu()])


def get_header(app):
    header = html.Div(
        [
            html.Div(
                [
                    html.Img(
                        src=app.get_asset_url("KTH.png"),
                        className="logo",
                        style={'height':'15%', 'width':'15%'}
                    ),
                    html.A(
                        html.Button("Search here", id="learn-more-button"),
                        href="https://www.google.com/", 
                    ),
                ],
                className="row",
            ),
            html.Div(
                [
                    html.Div(
                        [html.H5("Analysis and survey results")],
                        className="seven columns main-title",
                    ),
                    html.Div(
                        [
                            dcc.Link(
                                "Full View",
                                href="/dash-financial-report/full-view",
                                className="full-view-link",
                            )
                        ],
                        className="five columns",
                    ),
                ],
                className="twelve columns",
                style={"padding-left": "0"},
            ),
        ],
        className="row",
    )
    return header


def get_menu():
    menu = html.Div(
        [
            dcc.Link(
                "Overview",
                href="/DATA_ANALYSIS/overview",
                className="tab first",
            ),
            dcc.Link(
                "VARIABLES FOR THE MODEL",
                href="/DATA_ANALYSIS/graph_analysis",
                className="tab",
            ),
            dcc.Link(
                "NEW-RESULTAT-EXTERNAL",
                href="/DATA_ANALYSIS/correlation",
                className="tab",
            ),
            dcc.Link(
                "NEW-RESULTAT-INTERNAL",
                href="/DATA_ANALYSIS/indicators",
                className="tab",
            ),
      
            dcc.Link(
                "28-29",
                href="/DATA_ANALYSIS/documentation_tables",
                className="tab",
            ),

            dcc.Link(
                "28-29",
                href="/DATA_ANALYSIS/documentation_tables",
                className="tab",
            ),
            dcc.Link(
                "28-29",
                href="/DATA_ANALYSIS/documentation_tables",
                className="tab",
            ),
            dcc.Link(
                "28-29",
                href="/DATA_ANALYSIS/documentation_tables",
                className="tab",
            ),
            dcc.Link(
                "28-29",
                href="/DATA_ANALYSIS/documentation_tables",
                className="tab",
            ),
            dcc.Link(
                "28-29",
                href="/DATA_ANALYSIS/documentation_tables",
                className="tab",
            ),
            dcc.Link(
                "28-29",
                href="/DATA_ANALYSIS/documentation_tables",
                className="tab",
            ),

        ],
        className="row all-tabs",
    )
    return menu


def make_dash_table(df):
    """ Return a dash definition of an HTML table for a Pandas dataframe """
    table = []
    for index, row in df.iterrows():
        html_row = []
        for i in range(len(row)):
            html_row.append(html.Td([row[i]]))
        table.append(html.Tr(html_row))
    return table
