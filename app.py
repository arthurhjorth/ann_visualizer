from __future__ import annotations
from math import tanh
from nicegui import ui,  app
from random import uniform 
from math import tanh, sqrt  
from math import exp


TARGET_VECTOR = [-1.0, -1.0, 1.0, -1.0]  # one-hot for "time"
INPUT_TEXT = 'Once upon a'
OUTPUT_WORDS = ['newspaper', 'country', 'time', 'blue']
INPUT_WORDS = ['Once', 'upon', 'a', 'in']

NUM_LAYERS = 3
NEURONS_PER_LAYER = 4
LAYER_X = [60, 400, 740]
Y_POS = [60, 180, 300, 420]
NODE_BASE_SIZE = 22
OUT_MIN_SIZE, OUT_MAX_SIZE = 5, 70

INPUT_VECTOR = [1.0, 1.0, 1.0, 0.0]

def zero_params():
    weights = {(L, i, j): 0.0
               for L in range(NUM_LAYERS-1)
               for i in range(NEURONS_PER_LAYER)
               for j in range(NEURONS_PER_LAYER)}
    biases = {(L, j): 0.0
              for L in range(1, NUM_LAYERS)
              for j in range(NEURONS_PER_LAYER)}
    return weights, biases

weights, biases = zero_params()


def randomize_params():
    from random import uniform
    # randomize all weights
    for k in weights:
        weights[k] = uniform(-1.0, 1.0)
    # randomize all biases except input layer (layer 0 has none anyway)
    for k in biases:
        biases[k] = uniform(-1.0, 1.0)

    # clear selection and hide the slider
    current_selection.update({'type': None, 'key': None})
    try:
        slider.visible = False
        slider.update()
    except NameError:
        pass

    # clear chart and history
    global distance_history, step_idx
    distance_history = []
    step_idx = 0
    current_distance.text = '—'
    distance_chart.options['xAxis']['data'] = []
    distance_chart.options['series'][0]['data'] = []
    distance_chart.update()

    update_chart()

def softmax(xs):
    exps = [exp(x) for x in xs]
    s = sum(exps)
    return [v / s for v in exps] if s > 0 else [0.0] * len(xs)

def forward():
    a = [INPUT_VECTOR[:]]
    for L in range(1, NUM_LAYERS):
        prev, out = a[-1], []
        for j in range(NEURONS_PER_LAYER):
            s = biases[(L, j)]
            for i in range(NEURONS_PER_LAYER):
                s += weights[(L-1, i, j)] * prev[i]
            out.append(tanh(s))
        a.append(out)
    return a

def make_nodes_and_edges(activations):
    nodes, edges = [], []
    for L in range(NUM_LAYERS):
        for j in range(NEURONS_PER_LAYER):
            node_id = f'{L}-{j}'
            if L == 0:
                name = INPUT_WORDS[j]
            elif L < NUM_LAYERS - 1:
                name = f'L{L}·N{j}'
            else:
                name = OUTPUT_WORDS[j]
            size = OUT_MIN_SIZE + (activations[L][j] + 1) * 0.5 * (OUT_MAX_SIZE - OUT_MIN_SIZE)
            nodes.append({
                'id': node_id, 'name': name, 'x': LAYER_X[L], 'y': Y_POS[j],
                'symbolSize': size, 'value': round(activations[L][j], 3),
                'layer': L, 'index': j, 'bias': biases.get((L, j), 0.0),
                'label': {'show': True, 'position': 'right'},
                'itemStyle': {'borderColor': '#999', 'borderWidth': 1},
            })
    for L in range(NUM_LAYERS - 1):
        for i in range(NEURONS_PER_LAYER):
            for j in range(NEURONS_PER_LAYER):
                w = weights[(L, i, j)]
                edges.append({
                    'id': f'{L}-{i}-{j}',
                    'name': f'w(L{L}:{i}→L{L+1}:{j})',
                    'source': f'{L}-{i}', 'target': f'{L+1}-{j}',
                    'value': round(w, 3),
                    'lineStyle': {'width': 1 + abs(w) * 4,
                                  'color': '#3a7' if w >= 0 else '#c55',
                                  'opacity': 0.9},
                })
    return nodes, edges

with ui.row().classes('w-full items-start gap-6'):
    with ui.column().classes('min-w-[260px] max-w-[300px]'):
        ui.button('Start (randomize weights & biases)', on_click=randomize_params)\
            .props('color=primary').classes('w-full mb-2')

        ui.separator()
        ui.label('Distance to target word "time" (d=0 is perfect prediction)').classes('text-base font-medium')
        current_distance = ui.label('—').classes('text-sm text-gray-600')
        ui.label('This is also called the "loss"').classes('text-base font-medium')
        ui.label('We want to minimize that loss.').classes('text-base font-medium')

        ui.notify(f"current distance: {current_distance}")
        distance_chart = ui.echart({
            'animation': False,
            'grid': {'left': 30, 'right': 10, 'top': 40, 'bottom': 24},
            'xAxis': {'type': 'category', 'name': 'step', 'boundaryGap': False, 'axisLabel': {'fontSize': 10}},
            'yAxis': {'type': 'value', 'name': 'distance'},
            'series': [{'type': 'line', 'symbol': 'none', 'data': []}],
            'tooltip': {'trigger': 'axis'}
        }).classes('w-full h-[160px]')


    with ui.column().classes('grow'):
        ui.label(f"Completing '{INPUT_TEXT}...'").classes('text-2xl font-semibold')
        chart = ui.echart({'animation': False}).classes('w-[900px] h-[520px]')

        editor_card = ui.card().classes('w-[900px] mt-2')
        editor_title = ui.label('Select a neuron or connection').classes('text-base font-medium')
        editor_sub = ui.label('').classes('text-sm text-gray-600')

        # NOTE: handler is passed here (no .on_change method in 2.24)
        def on_slider_change(e):
            # clamp then quantize to nearest 0.1
            v = max(-1.0, min(1.0, float(e.value)))
            v = round(v * 10) / 10.0

            stype, key = current_selection['type'], current_selection['key']
            if not stype or not key:
                return

            if stype == 'bias':
                biases[key] = v
            else:
                weights[key] = v

            # reflect quantized value in the slider and editor text
            slider.value = v
            slider.update()
            editor_sub.text = f'Current value: {v:+.1f}  •  Drag to adjust in [-1, 1] (step 0.1)'

            update_chart()

        slider = ui.slider(min=-1, max=1, step=0.1, on_change=on_slider_change).classes('w-full')
        slider.visible = False
    with ui.column().classes('min-w-[260px] max-w-[300px]'):
        ui.label('Completions').classes('text-lg font-medium')
        completions_md = ui.markdown('').classes('text-sm')

current_selection = {'type': None, 'key': None}

distance_history = []
step_idx = 0

def distance_to_target(outs):
    # Euclidean distance to the one-hot target
    return sqrt(sum((outs[j] - TARGET_VECTOR[j])**2 for j in range(NEURONS_PER_LAYER)))

def update_chart():
    global step_idx
    acts = forward()
    nodes, edges = make_nodes_and_edges(acts)
    option = {
        'tooltip': {'trigger': 'item'},
        'series': [{
            'type': 'graph',
            'layout': 'none',
            'roam': False,
            'draggable': False,
            'data': nodes,
            'links': edges,
            'edgeSymbol': ['none', 'arrow'],
            'edgeSymbolSize': 8,
            'emphasis': {'focus': 'adjacency'},

            # Node labels: show name and activation
            'label': {
                'show': True,
                'position': 'inside',
                ':formatter': (
                    "function(p){"
                    "  var v = Number(p.value);"
                    "  var s = (v>=0?'+':'') + v.toFixed(2);"
                    "  return p.data.name + '\\n' + s;"
                    "}"
                ),
                'fontSize': 10,
                'lineHeight': 14
            },

            # Edge labels: show weight in the middle
            'edgeLabel': {
                'show': True,
                'position': 'middle',
                ':formatter': (
                    "function(p){"
                    "  var v = (p.data && p.data.value != null) ? Number(p.data.value) : Number(p.value);"
                    "  if (isNaN(v)) return '';"
                    "  return (v>=0?'+':'') + v.toFixed(2);"
                    "}"
                ),
                'fontSize': 9
            },
        }],
    }
    chart.options.clear()
    chart.options.update(option) 
    chart.update()

    # distance 
    outs = acts[-1]
    d = distance_to_target(outs)
    current_distance.text = f'd = {d:.4f}'

    step_idx += 1
    distance_history.append((step_idx, d))
    # keep last N points to avoid crazy growth
    if len(distance_history) > 300:
        del distance_history[:len(distance_history)-300]

    x_data = [str(s) for s, _ in distance_history]
    y_data = [val for _, val in distance_history]
    distance_chart.options['xAxis']['data'] = x_data
    distance_chart.options['series'][0]['data'] = y_data
    distance_chart.update()
    probs = softmax(outs)
    text = '\n'.join(
        f'- Once upon a **{OUTPUT_WORDS[j]}** {probs[j]*100:.1f}%'
        for j in range(len(OUTPUT_WORDS))
    )
    completions_md.set_content(text)


def select_bias(layer: int, index: int):
    if layer == 0:
        ui.notify('Input neurons have no bias.', type='warning')
        return
    current_selection.update({'type': 'bias', 'key': (layer, index)})
    editor_title.text = f'Editing bias: L{layer} · N{index}'
    editor_sub.text = 'Drag the slider to adjust the bias in [-1, 1]'
    slider.value = float(biases[(layer, index)])
    slider.visible = True
    slider.update()

def select_weight(L: int, i: int, j: int):
    current_selection.update({'type': 'weight', 'key': (L, i, j)})
    editor_title.text = f'Editing weight: L{L}·N{i} → L{L+1}·N{j}'
    editor_sub.text = 'Drag the slider to adjust the weight in [-1, 1]'
    slider.value = float(weights[(L, i, j)])
    slider.visible = True
    slider.update()

def on_chart_click(evt: dict):
    dt = evt.args.get('dataType')
    data = evt.args.get('data', {})
    if dt == 'node':
        # ui.notify('found node')
        select_bias(int(data['layer']), int(data['index']))
    elif dt == 'edge':
        src, tgt = str(data.get('source')), str(data.get('target'))
        L, i = map(int, src.split('-')); L2, j = map(int, tgt.split('-'))
        if L2 == L + 1:
            select_weight(L, i, j)

chart.on('chart:click', on_chart_click, args=['dataType', 'data'])

with ui.header().classes('items-center'):
    ui.label('ANN to predict next word').classes('text-lg font-medium')
with ui.footer():
    ui.label('Click node (bias) or arrow (weight); outputs resize with value.')

update_chart()
ui.run(reload=False)
