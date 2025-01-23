import streamlit as st
import plotly.graph_objs as go

import aqua_blue


def predict_next_steps(
        series: aqua_blue.TimeSeries,
        horizon: int,
        reservoir_dimensionality: int,
        regularization_parameter: float
) -> aqua_blue.TimeSeries:

    computer = aqua_blue.EchoStateNetwork(
        input_dimensionality=series.num_dims,
        reservoir_dimensionality=reservoir_dimensionality,
        regularization_parameter=regularization_parameter
    )
    computer.train(series)

    return computer.predict(horizon=horizon)


def format_with_superscript(power):
    superscripts = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    return f"10{str(power).translate(superscripts)}"


def main():
    st.title("Apollo-Lite 🔭 🤖", help="Time series prediction")

    # File upload
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        # Read the CSV
        column_names = uploaded_file.readline().decode("utf-8").strip("\n").split(",")[1:]
        series = aqua_blue.TimeSeries.from_csv(uploaded_file)

        # display as dataframe
        data = {"time": series.times}
        for name, column in zip(column_names, series.dependent_variable.T):
            data[name] = column

        st.dataframe(data, use_container_width=True, hide_index=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            n_steps = st.number_input(
                "Number of steps to predict",
                min_value=1,
                value=10,
                help="How many steps do you want to predict into the future?"
            )
        with col2:
            reservoir_dimensionality = st.number_input(
                "Dimensionality of reservoir",
                min_value=1,
                value=100,
                help="How large should the reservoir be? Larger means better predictions, but will be slower"
            )
        with col3:
            regularization_power = int(st.select_slider(
                "Regularization parameter",
                options=list(range(-10, -1)),
                format_func=format_with_superscript
            ))

        # Predict and display
        if st.button("Predict"):

            fig = go.Figure()

            for column_name, y in zip(column_names, series.dependent_variable.T):
                fig.add_trace(
                    go.Scatter(x=series.times, y=y, name=column_name, mode="lines")
                )

            regularization_parameter = 10 ** regularization_power
            predictions = predict_next_steps(series, n_steps, reservoir_dimensionality, regularization_parameter)

            for column_name, y in zip(column_names, predictions.dependent_variable.T):
                fig.add_trace(
                    go.Scatter(x=predictions.times, y=y, name=f"{column_name} predicted",
                               line={"dash": "dash"}, mode="lines")
                )

            fig.update_layout(
                title="Time Series Prediction",
                xaxis_title="time",
                yaxis_title="time series"
            )
            tab1, tab2 = st.tabs(["Chart 📈", "Table 📑"])
            with tab1:
                st.plotly_chart(fig, theme="streamlit", use_container_width=True)

            with tab2:
                data = {"time": predictions.times}
                for name, column in zip(column_names, predictions.dependent_variable.T):
                    data[name] = column

                st.dataframe(data, use_container_width=True, hide_index=True)


if __name__ == "__main__":

    main()
