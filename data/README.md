# Dataset Instructions

Place a CSV file named `spam.csv` in this folder before training or running the app.

Expected format:

- one column containing the message text
- one column containing the class label

Common supported column names:

- text / message / sms / content / body
- label / target / spam / class

Example:

| text | label |
| --- | --- |
| Congratulations! You have won a free prize. | spam |
| Hello, are we still meeting at 5 PM? | ham |

The loader is flexible and will accept common variations, but labels should ultimately be normalized to `spam` or `ham`.
