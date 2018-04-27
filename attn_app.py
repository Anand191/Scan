import numpy as np
from bokeh.plotting import figure
from bokeh.models import LinearColorMapper, BasicTicker, ColorBar, TextInput, FuncTickFormatter, ColumnDataSource
from bokeh.io import curdoc
from bokeh.layouts import row,layout, widgetbox
from readData import search_class, gen_parts, gen_attn

def create_source(new):
    tags = search_class(new)
    sents, sent_tags, sent_idx = gen_parts(new.split(' '), tags)
    src = gen_attn(new, sents, sent_tags, sent_idx)
    data_dict = {'image':[np.flip(src, axis=0)]}
    return (ColumnDataSource(data=data_dict))

def create_plot(source, labels):
    color_mapper = LinearColorMapper(palette="Viridis256", low=0.0, high=1.0)
    color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(desired_num_ticks=2),
                         label_standoff=12, border_line_color=None, location=(1, 1))
    plot = figure(x_range=(0, 1), y_range=(0, 1), x_axis_location="above")
    plot.image(image='image',source=source, x=0, y=0, dw=1, dh=1, palette="Viridis256")
    plot.text(x='x', y='y', text='text', source=labels, angle=1.5, text_color="#000000")
    plot.axis.visible=False
    plot.add_layout(color_bar, 'right')
    return plot

def create_labels(new):
    words = new.split(' ')
    x = np.linspace(0.035,0.96, num=len(words))
    y = np.repeat(0.5,len(words))
    data_dict = {'x':x, 'y':y, 'text':words}
    return (ColumnDataSource(data=data_dict))


def my_text_input_handler(attr, old, new):
    print("Previous label: " + old)
    print("Updated label: " + new)
    src = create_source(new)
    labs = create_labels(new)
    source.data.update(src.data)
    labels.data.update(labs.data)
    #print(source.data)

source = create_source('turn left')
labels = create_labels('turn left')

text_input = TextInput(value="turn left", title="New SCAN Command:")
text_input.on_change("value", my_text_input_handler)

plot = create_plot(source, labels)

layout = layout([
                [widgetbox(text_input)],
                [plot]
                ])

curdoc().add_root(layout)