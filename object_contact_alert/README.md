# Object Contact Alert

This feature triggers an alert when a person is close to or touching a selected dangerous object.

For the full demo with both features together, run the combined monitor from the repository root:

```bash
python combined_monitor.py
```

Open:

```text
http://127.0.0.1:5004
```

## Standalone Run

Use this when you only want to test the object contact feature.

```bash
cd object_contact_alert/app
python object_contact_web.py
```

Open:

```text
http://127.0.0.1:5003
```

If you run from a different location, make sure `yolov8x-seg.pt` and `yolov8x-worldv2.pt` are available as expected by the script configuration.

## File Map

```text
object_contact_alert/
├── README.md
├── app/
│   ├── object_contact_core.py        # model loading + contact detection logic
│   ├── object_contact_web.py         # standalone Flask UI, port 5003
│   └── object_contact_core.ipynb
└── benchmark/
    ├── benchmark.py                  # benchmark each danger class and distance
    └── benchmark_annotated.ipynb
```

## Web UI Features

- Live camera stream in the browser.
- Debug view with masks, labels, boxes, and buffer boxes.
- Clean view that hides model overlays unless an alert is active.
- Checkbox list for selecting monitored object classes.
- Small / Medium / Large contact buffer selector.
- Status panel for detected people, detected danger items, and active alerts.

## Detection Logic

This feature uses two model sources:

```text
YOLOv8x-seg:
    person + COCO danger classes
    outputs segmentation masks and boxes

YOLOWorld:
    custom danger classes
    outputs boxes only
```

Contact checks:

```text
COCO danger object:
    Priority 1: person mask + danger mask
    Priority 2: person mask + expanded danger box
    Priority 3: person box + expanded danger box

YOLOWorld custom object:
    Priority 1: person mask + expanded danger box
    Priority 2: person box + expanded danger box
```

## Dangerous Objects

COCO classes:

```text
knife
scissors
bottle
wine glass
fork
microwave
oven
toaster
```

YOLOWorld custom classes:

```text
cigarette lighter
box of matches
prescription medicine bottle
electrical extension cord
plastic grocery bag
wax candle
clothes iron
medical syringe
household cleaning spray bottle
```

## Benchmark Tool

Run:

```bash
cd object_contact_alert/benchmark
python benchmark.py
```

Other modes:

```bash
python benchmark.py --report
python benchmark.py --fp
```

The benchmark records detection rate and false positive behavior across object classes and distances.

## Deployment

See the root deployment guide:

```text
../DEPLOYMENT.md
```
