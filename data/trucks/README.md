# Truck specifications

Add truck makes, models, sleeper dimensions, and fit notes. Supported formats: **CSV** or **JSON**.

## Recommended CSV columns

| Column | Description |
|--------|-------------|
| make | Truck make (e.g. Freightliner, Kenworth) |
| model | Model name |
| sleeper_type | e.g. Cab-over, conventional, flat top |
| sleeper_length_in | Sleeper length in inches |
| sleeper_width_in | Sleeper width in inches |
| mattress_size_fits | e.g. Short Queen, Twin XL, Custom |
| fit_notes | Any special fit or installation notes |

You can add more columns; all will be searchable.

## JSON format

Array of truck objects, or single object. Same keys as above.

Example: see `trucks_sample.csv` (replace with your real file when you have it).
