# Mattress catalog

Add your full catalog here. Supported formats: **CSV** or **JSON**.

## Recommended CSV columns

| Column | Description |
|--------|-------------|
| model | Model name / SKU |
| name | Display name |
| sizes | e.g. Twin, Full, Queen, King (or list in separate rows) |
| price | Price or price range |
| materials | Foam, innerspring, hybrid, etc. |
| thickness | Thickness / height |
| recommendations | Who it's best for (e.g. side sleepers, truck sleepers) |
| description | Short product description |

You can add more columns; all will be searchable.

## JSON format

Either an array of products, or a single object. Each product should have keys like: `model`, `name`, `sizes`, `price`, `materials`, `recommendations`, `description`.

Example: see `mattress_catalog_sample.csv` (replace with your real file when you have it).
