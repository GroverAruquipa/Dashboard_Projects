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
                        src=app.get_asset_url("dash-financial-logo.png"),
                        className="logo",
                    ),
                    html.A(
                        html.Button("Learn More", id="learn-more-button"),
                        href="https://www.forex.com/en/", 
                    ),
                ],
                className="row",
            ),
            html.Div(
                [
                    html.Div(
                        [html.H5("FOREX PRICES COMPARATION")],
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
                href="/FOREX1-DASH/overview",
                className="tab first",
            ),
            dcc.Link(
                "graph_analysis",
                href="/FOREX1-DASH/graph_analysis",
                className="tab",
            ),
            dcc.Link(
                "correlation",
                href="/FOREX1-DASH/correlation",
                className="tab",
            ),
            dcc.Link(
                "Data table",
                href="/FOREX1-DASH/indicators",
                className="tab",
            ),
      
            dcc.Link(
                "documentation_tables",
                href="/FOREX1-DASH/documentation_tables",
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
