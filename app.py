import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression


# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------

st.set_page_config(
    page_title="Student Lifestyle Explorer",
    layout="wide"
)

st.title("Student Lifestyle Explorer")

st.caption(
    "Upload a student lifestyle dataset, explore the data, "
    "and use linear regression to predict GPA."
)


# ---------------------------------------------------------
# Upload CSV file
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type="csv"
)


if uploaded_file:

    # Read the uploaded CSV
    df = pd.read_csv(uploaded_file)


    # -----------------------------------------------------
    # Data preview
    # -----------------------------------------------------

    st.header("Data Preview")

    st.write(
        f"The dataset contains {df.shape[0]:,} rows "
        f"and {df.shape[1]} columns."
    )

    st.dataframe(
        df.head(10),
        hide_index=True,
        width="stretch"
    )


    # -----------------------------------------------------
    # Find numeric columns
    # -----------------------------------------------------

    numeric_cols = df.select_dtypes(
        include="number"
    ).columns.tolist()

    # Remove Student_ID because it is an identifier,
    # not a meaningful variable for analysis.
    if "Student_ID" in numeric_cols:
        numeric_cols.remove("Student_ID")


    if len(numeric_cols) < 3:

        st.error(
            "This dataset needs at least three numeric "
            "columns for the analysis."
        )

        st.stop()


    # -----------------------------------------------------
    # Chart section
    # -----------------------------------------------------

    st.header("Explore the Data")

    chart_type = st.selectbox(
        "Choose a chart type",
        ["Line", "Scatter", "Histogram"]
    )


    # Choose the main column
    selected_column = st.selectbox(
        "Choose a column",
        numeric_cols
    )


    # -----------------------------------------------------
    # Line chart
    # -----------------------------------------------------

    if chart_type == "Line":

        fig = px.line(
            df,
            x="Student_ID" if "Student_ID" in df.columns else df.index,
            y=selected_column,
            title=f"{selected_column} by Student"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )


    # -----------------------------------------------------
    # Scatter plot
    # -----------------------------------------------------

    elif chart_type == "Scatter":

        # Create a list of columns other than the
        # selected column
        scatter_columns = [
            col for col in numeric_cols
            if col != selected_column
        ]

        second_column = st.selectbox(
            "Choose the second column for the scatter plot",
            scatter_columns
        )

        fig = px.scatter(
            df,
            x=selected_column,
            y=second_column,
            title=f"{second_column} vs. {selected_column}"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )


    # -----------------------------------------------------
    # Histogram
    # -----------------------------------------------------

    elif chart_type == "Histogram":

        fig = px.histogram(
            df,
            x=selected_column,
            title=f"Distribution of {selected_column}"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )


    # -----------------------------------------------------
    # Linear Regression
    # -----------------------------------------------------

    st.header("Predict GPA with Linear Regression")

    st.write(
        "Choose two numeric variables as inputs and a target "
        "variable. The model will use the selected inputs "
        "to predict the target."
    )


    # Input variable 1
    input_1 = st.selectbox(
        "Choose the first input",
        numeric_cols,
        index=numeric_cols.index("Study_Hours_Per_Day")
        if "Study_Hours_Per_Day" in numeric_cols else 0
    )


    # Input variable 2
    available_input_2 = [
        col for col in numeric_cols
        if col != input_1
    ]

    input_2 = st.selectbox(
        "Choose the second input",
        available_input_2,
        index=available_input_2.index("Sleep_Hours_Per_Day")
        if "Sleep_Hours_Per_Day" in available_input_2
        else 0
    )


    # Target variable
    available_targets = [
        col for col in numeric_cols
        if col != input_1 and col != input_2
    ]

    target = st.selectbox(
        "Choose the target",
        available_targets,
        index=available_targets.index("GPA")
        if "GPA" in available_targets
        else 0
    )


    # -----------------------------------------------------
    # Fit the model
    # -----------------------------------------------------

    X = df[[input_1, input_2]]
    y = df[target]

    # Remove rows with missing values
    model_data = pd.concat(
        [X, y],
        axis=1
    ).dropna()

    X_clean = model_data[[input_1, input_2]]
    y_clean = model_data[target]

    model = LinearRegression()

    model.fit(
        X_clean,
        y_clean
    )


    # -----------------------------------------------------
    # Prediction inputs
    # -----------------------------------------------------

    st.subheader("Make a Prediction")

    st.write(
        f"Enter values for {input_1} and {input_2} "
        f"to predict {target}."
    )


    value_1 = st.number_input(
        f"{input_1}",
        value=float(df[input_1].mean())
    )


    value_2 = st.number_input(
        f"{input_2}",
        value=float(df[input_2].mean())
    )


    # IMPORTANT:
    # The prediction uses the exact same column names
    # and order as the training data.
    prediction_data = pd.DataFrame(
        {
            input_1: [value_1],
            input_2: [value_2]
        }
    )


    prediction = model.predict(
        prediction_data
    )[0]


    # Display prediction
    st.metric(
        label=f"Predicted {target}",
        value=f"{prediction:.2f}"
    )


    # -----------------------------------------------------
    # Explain the model
    # -----------------------------------------------------

    st.info(
        "This model is used to predict a student's GPA "
        "based on two selected lifestyle variables; when "
        "it is wrong, the predicted GPA may overestimate "
        "or underestimate the student's actual GPA."
    )


    # -----------------------------------------------------
    # Show model information
    # -----------------------------------------------------

    with st.expander("See model details"):

        st.write(
            f"**Input variables:** {input_1}, {input_2}"
        )

        st.write(
            f"**Target variable:** {target}"
        )

        st.write(
            f"**Number of observations used:** "
            f"{len(model_data):,}"
        )

        st.write(
            f"**Model intercept:** "
            f"{model.intercept_:.4f}"
        )

        coefficients = pd.DataFrame(
            {
                "Variable": [input_1, input_2],
                "Coefficient": [
                    model.coef_[0],
                    model.coef_[1]
                ]
            }
        )

        st.dataframe(
            coefficients,
            hide_index=True,
            width="stretch"
        )


    # -----------------------------------------------------
    # Raw data
    # -----------------------------------------------------

    with st.expander("See raw data"):

        st.dataframe(
            df,
            hide_index=True,
            width="stretch"
        )


else:

    st.info(
        "Upload a CSV file to get started. "
        "The file should contain at least three numeric columns."
    )
