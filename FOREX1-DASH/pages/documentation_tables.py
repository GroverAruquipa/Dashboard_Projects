import dash_html_components as html
from utils import Header


def create_layout(app):
    return html.Div(
        [
            Header(app),
            # page 6
            html.Div(
                [
                    # Row 1
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6("COMMENTS", className="subtitle padded"),
                                    html.Br([]),
                                    html.Div(
                                        [
                                            html.P(
                                                "THIS IS A TEST EXAMPLE OF dASHBOARD USING THE PYTHON APPI DASH"
                                            ),
                                            html.P(
                                                "The data or dataset was extracted from Yahoo finannce"
                                            ),
                                        ],
                                        style={"color": "#7a7a7a"},
                                    ),
                                ],
                                className="row",
                            ),
                            html.Div(
                                [
                                    html.H6("Resume", className="subtitle padded"),
                                    html.Br([]),
                                    html.Div(
                                        [
                                            html.Li("Divisa JPY."),
                                            html.Li(
                                                "According to the movement of the market in recent years it is observed that this currency had variable pivot points, yet the last yeas remains stable with respect to its history.*"
                                            ),
                                            html.Li(
                                                "It is recommended to review the attached repositories to review the approximation algorithms using Deep QLearning"
                                            ),
                                        ],
                                        id="reviews-bullet-pts",
                                    ),
                                    
                                ],
                                className="row",
                            ),
                        ],
                        className="row ",
                    )
                ],
                className="sub_page",
            ),
        ],
        className="page",
    )
