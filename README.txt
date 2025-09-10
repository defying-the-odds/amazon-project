This is an Amazon Product Recommender. 

1.Downlaod data: https://drive.google.com/file/d/1wdnaxYtaj_35WolDL0ibq3EZy2dv7y9B/view?usp=sharing

2. run setup_database.py

3. Set up virtual enviorment
  a.python3 -m venv myenv
  b.source myenv/bin/activate (MacOS/Linux) myenv\Scripts\activate (Windows)
  c. Install dependencies (found in Requriements.txt) use: pip install package_name
  d. use deactivate to close out of virtual enviorment

4. To run virtual enviorment you will need 2 seperate terminals:
  a. streamlit run app.py (runs the streamlit part)
  b. python main.py (runs the backend)
