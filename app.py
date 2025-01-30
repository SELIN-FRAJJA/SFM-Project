from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import numpy as np

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def welcome():
    if request.method == "POST":
        age = request.form["age"]
        sex = request.form["sex"]

        # Save age and sex to a text file
        with open("user_data.txt", "w") as file:
            file.write(f"Age: {age}\nSex: {sex}")

        # Redirect to the health options page
        return redirect(url_for("health_options"))

    return render_template("welcome.html")


@app.route("/health-options", methods=["GET", "POST"])
def health_options():
    if request.method == "POST":
        option = request.form["option"]

        # Check if the selected option is "skin"
        if option.lower() == "skin":
            # Redirect to the Skinapp.py Flask application
            return redirect("http://127.0.0.1:5001")  # Adjust the URL and port if needed

        # Check if the selected option is "fever"
        elif option.lower() == "fever":
            # Redirect to the Feverapp.py Flask application
            return redirect("http://localhost:8501")  # Adjust the URL and port if needed

        # Check if the selected option is "mental health"
        elif option.lower() == "maternal health":
            # Redirect to the mhapp.py Flask application
            return redirect("http://127.0.0.1:5003")  # Adjust the URL and port if needed

        # For other options, display the selected value
        return f"You selected: {option}"

    return render_template("health_options.html")

@app.route("/map")
def map_page():
    # This route renders mapindex.html
    return render_template("mapindex.html")

if __name__ == "__main__":
    app.run(debug=True)
