import pandas as pd
import typing as tp
from MarketStructure import MRP, MKS
from dataclasses import dataclass, field
import numpy as np
from collections import namedtuple
from bokeh.transform import factor_cmap
from bokeh.models import Slider, DatetimeTickFormatter, CustomJS, HoverTool, ColumnDataSource, Button, Div, Span
from bokeh.plotting import figure, column, row, show

DateIndexValueTuple = namedtuple("DateIndexValueTuple", ["date_index", "value"])


def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)


@dataclass
class ChartComponents:
    highs_normal: tp.List[MKS.Peak] = field(default_factory=list)
    lows_normal: tp.List[MKS.Trough] = field(default_factory=list)
    highs_temp: tp.List[MKS.Peak] = field(default_factory=list)
    lows_temp: tp.List[MKS.Trough] = field(default_factory=list)
    highs_is_cms: tp.List[MKS.Peak] = field(default_factory=list)
    lows_is_cms: tp.List[MKS.Trough] = field(default_factory=list)
    highs_was_cms: tp.List[MKS.Peak] = field(default_factory=list)
    lows_was_cms: tp.List[MKS.Trough] = field(default_factory=list)


default_factory_DIVT = lambda: DateIndexValueTuple(date_index=np.array([]), value=np.array([]))


def split_nodes(node: tp.Union[MKS.Peak, MKS.Trough], chart_components: ChartComponents):
    if isinstance(node, MKS.Peak):
        if node.is_cms:
            chart_components.highs_is_cms.append(node)
        elif node.was_cms:
            chart_components.highs_was_cms.append(node)
        elif node.is_permanent:
            chart_components.highs_normal.append(node)
        else:
            chart_components.highs_temp.append(node)
    else:
        if node.is_cms:
            chart_components.lows_is_cms.append(node)
        elif node.was_cms:
            chart_components.lows_was_cms.append(node)
        elif node.is_permanent:
            chart_components.lows_normal.append(node)
        else:
            chart_components.lows_temp.append(node)


def clean_monotonic_stack_of_nodes(stack: tp.List[MKS.Node], earliest_node: MKS.Node):
    """
    Removes nodes from the stack that are not monotonic with the earliest node

    :param stack:
    :param earliest_node:
    :return:
    """

    while len(stack) > 0:

        if stack[-1].date_index > earliest_node.date_index:
            stack.pop()
        elif stack[-1].date_index == earliest_node.date_index:
            if isinstance(stack[-1], type(earliest_node)):
                stack.pop()
            else:
                if stack[-1].prev and stack[-1].prev.date_index == earliest_node.date_index:
                    # earliest_node occurs before stack[-1] so remove
                    stack.pop()
                else:
                    break
        else:
            break

    return stack


def create_chart_components(snapshot: MKS.MarketStructure) -> ChartComponents:
    chart_components = ChartComponents()
    for a_node in snapshot:
        split_nodes(node=a_node, chart_components=chart_components)
    return chart_components


def update_chart_components(new_nodes: MKS.MarketStructure, chart_components: ChartComponents):
    attr = [attr_name for attr_name in dir(chart_components) if not attr_name.startswith("_")]
    for attr_name in attr:
        setattr(chart_components, attr_name, clean_monotonic_stack_of_nodes(stack=getattr(chart_components, attr_name),
                                                                            earliest_node=new_nodes.head))

    for a_node in new_nodes:
        split_nodes(node=a_node, chart_components=chart_components)

    return chart_components


def transform_component_to_data_source(chart_components: ChartComponents):
    date_index = []
    value = []
    attribute = []
    trend = []
    comment = []
    furthest_date = None
    for attr_name in attr:
        for node in getattr(chart_components, attr_name):
            date_index.append(node.date_index)
            value.append(node.value)
            attribute.append(attr_name)
            trend.append(node.trend.upper() if node.trend else None)
            comment.append(node.COMMENT)

    sorting_idx = argsort(date_index)
    date_index = np.array(date_index)[sorting_idx]
    furthest_date = date_index[-1]
    value = np.array(value)[sorting_idx]
    attribute = np.array(attribute)[sorting_idx]
    trend = np.array(trend)[sorting_idx]
    comment = np.array(comment)[sorting_idx]
    curr_trend = trend[-2] if trend[-1] == None else trend[-1]
    curr_comment = comment[-2] if comment[-1] == "" else comment[-1]

    return ColumnDataSource(
        data=dict(date_index=date_index, value=value, attribute=attribute)), curr_trend, curr_comment, furthest_date


def compute_sources(animation_list):
    data_sources = []
    trend = []
    comment = []
    chart_components = None
    furthest_dates = []
    for i, nodes in enumerate(animation_list):
        if i == 0:
            chart_components = create_chart_components(nodes.market_structure)
            out = transform_component_to_data_source(chart_components)
            furthest_dates.append(out[3])
        else:
            chart_components = update_chart_components(nodes.market_structure, chart_components)
            out = transform_component_to_data_source(chart_components)
            furthest_dates.append(max(out[3], furthest_dates[-1]))

        data_sources.append(out[0])
        trend.append(out[1])
        comment.append(out[2])
    # print(type(furthest_dates[0].get_timestamp()))
    return data_sources, trend, comment, furthest_dates


def calc_bar_width(granularity, period_in_seconds):
    return period_in_seconds * 1000 / 2


map_attr_to_color = {'highs_normal': 'red', 'lows_normal': 'blue', 'highs_temp': 'pink', 'lows_temp': 'powderblue',
                     'highs_is_cms': 'green', 'lows_is_cms': 'black', 'highs_was_cms': 'purple',
                     'lows_was_cms': 'maroon'}

attr = ["highs_normal", "lows_normal", "highs_temp", "lows_temp", "highs_is_cms", "lows_is_cms", "highs_was_cms",
        "lows_was_cms"]
c_map = factor_cmap("attribute", palette=list(map_attr_to_color.values()), factors=attr)

hover = HoverTool(
    tooltips=[
        ('Date', '@date{%m-%d-%Y %H:%M}'),
        ('Open', '@open'),
        ('High', '@high'),
        ('Low', '@low'),
        ('Close', '@close'),
        ('Volume', '@volume'),
    ],

    formatters={
        '@date': 'datetime'
    },
    mode='mouse'
)


def prepare_oanda_data(data):
    prepped_data = data.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close"})
    prepped_data["date"] = prepped_data.index
    return prepped_data


def create_high_and_low_traces(fig, sources, current_source, trends, comments, furthest_dates):
    fig.circle(x="date_index", y="value", size=5, color=c_map, source=current_source, legend_field="attribute")
    for i, comment in enumerate(comments):
        comments[i] = '\n' + comment
    # Slider to control the animation state
    state = Slider(start=0, end=len(sources) - 1, value=0, step=1, title="State")

    # Play/Pause button
    play_button = Button(label="► Play", button_type="success")

    # Div to display the current trend
    trend_div = Div(text=f"<b>Current Trend: {trends[0]}</b>", width=200, height=50)

    comments_div = Div(text=f"<b>Market Structure Comment: {comments[0]}</b>", width=200, height=50)

    furthest_date_line = Span(location=pd.to_datetime(furthest_dates[0]).timestamp() * 1000,
                              dimension='height', line_color='black', line_dash='dashed', line_width=1)
    fig.add_layout(furthest_date_line)

    # Callback to update the chart and trend when the slider value changes
    callback = CustomJS(
        args=dict(sources=sources, state=state, current_source=current_source, trend_div=trend_div, trends=trends,
                  comments_div=comments_div, comments=comments,
                  fig=fig, furthest_dates=furthest_dates, furthest_date_line=furthest_date_line),
        code="""
            current_source.data = sources[state.value].data;
            trend_div.text = "<b>Current Trend: " + trends[state.value] + "</b>";
            comments_div.text = "<b>Market Structure Comment: " + comments[state.value] + "</b>";

            if (trends[state.value] == 'UPTREND') {
                fig.background_fill_color = "#E0F7FA";
            } else {
                fig.background_fill_color = "#ECEFF1";
            }
            
            let furthestDate = new Date(furthest_dates[state.value]);
            furthest_date_line.location = furthestDate.getTime();
        """)

    state.js_on_change('value', callback)

    # Play/Pause functionality, also updates the trend
    play_callback = CustomJS(
        args=dict(state=state, button=play_button, current_source=current_source, sources=sources, trend_div=trend_div,
                  trends=trends, comments_div=comments_div, comments=comments, fig=fig),
        code="""
        var isPlaying = false;
        var intervalId = null;

        if (button.intervalId !== undefined) {
            isPlaying = true;
        }

        function play() {
            isPlaying = true;
            button.label = "❚❚ Pause";
            button.intervalId = setInterval(function() {
                if (state.value < state.end) {
                    state.value = state.value + 1;
                } else {
                    state.value = state.start; // Restart from the beginning
                }
                current_source.data = sources[state.value].data;
                trend_div.text = "<b>Current Trend: " + trends[state.value] + "</b>";  // Update trend
                comments_div.text = "<b>Market Structure Comment: " + comments[state.value] + "</b>";  // Update comments
                
                if (trends[state.value] == 'UPTREND') {
                    fig.background_fill_color = "#E0F7FA";
                } else {
                    fig.background_fill_color = "#ECEFF1";
                }
            }, 500);  // Adjust the interval (milliseconds) for speed of animation
        }

        function pause() {
            isPlaying = false;
            button.label = "► Play";
            clearInterval(button.intervalId);
            button.intervalId = undefined;
        }

        // Toggle play/pause
        if (!isPlaying) {
            play();
        } else {
            pause();
        }
    """)

    play_button.js_on_click(play_callback)

    fig.add_tools(hover)

    return state, play_button, trend_div, comments_div, furthest_date_line


def animate_market_structure(data: pd.DataFrame, granularity: str, no_of_seconds_per_period: int,
                             animation_list: tp.List[MRP.AnimationListTuple]):
    """

    Animate the market structure

    :param data:
    :param granularity:
    :param no_of_seconds_per_period:
    :param animation_list:
    :return:
    """

    sources, trends, comments, furthest_dates = compute_sources(animation_list)

    data["date"] = data.index

    inc = data.close >= data.open
    dec = data.open > data.close
    w = calc_bar_width(granularity=granularity, period_in_seconds=no_of_seconds_per_period)

    current_source = sources[0]
    fig = figure(x_axis_type="datetime")
    # Actual Candlesticks
    fig.segment("date", "high", "date", "low", color="black", source=data)
    fig.vbar("date", w, "open", "close", fill_color="lawngreen", line_width=0,
             source=data[inc])
    fig.vbar("date", w, "open", "close", fill_color="tomato", line_width=0,
             source=data[dec])

    state, play_button, trend_div, comments_div, furthest_date_line = create_high_and_low_traces(
        fig=fig,
        sources=sources,
        current_source=current_source,
        trends=trends,
        comments=comments,
        furthest_dates=furthest_dates)
    fig.xaxis.axis_label = "Date"
    fig.yaxis.axis_label = "Price "
    fig.legend.location = "top_left"

    fig.xaxis.formatter = DatetimeTickFormatter(days="%m-%d-%Y")
    fig.sizing_mode = "stretch_both"

    layout = row(fig, column(state, play_button, trend_div, comments_div))
    layout.sizing_mode = "stretch_both"
    show(layout)
