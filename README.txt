This is an Amazon Product Recommender. 

1. run setup_database.py

2. Set up virtual enviorment
  a.python3 -m venv myenv
  b.source myenv/bin/activate (MacOS/Linux) myenv\Scripts\activate (Windows)
  c. Install dependencies (found in Requriements.txt) use: pip install package_name
  d. use deactivate to close out of virtual enviorment

3. To run virtual enviorment you will need 2 seperate terminals:
  a. streamlit run app.py (runs the streamlit part)
  b. python main.py (runs the backend)
