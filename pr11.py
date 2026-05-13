# ---------------------------------------------------------
# BIG DATA CONCEPT:
# Weather Data Analysis with Line Chart Visualization
# ---------------------------------------------------------

import matplotlib.pyplot as plt

def calculate_averages(filename):

    # Lists for storing data
    dates = []
    temperatures = []
    dew_points = []
    wind_speeds = []

    # Variables for averages
    total_temp = 0
    total_dew = 0
    total_wind = 0
    count = 0

    # Open file
    with open(filename, 'r') as file:

        # Skip header
        next(file)

        # Read each line
        for line in file:

            parts = line.split()

            # Extract values
            date = parts[0]
            temp = float(parts[1])
            dew = float(parts[2])
            wind = float(parts[3])

            # Store data in lists
            dates.append(date)
            temperatures.append(temp)
            dew_points.append(dew)
            wind_speeds.append(wind)

            # Add totals
            total_temp += temp
            total_dew += dew
            total_wind += wind

            count += 1

    # Calculate averages
    avg_temp = total_temp / count
    avg_dew = total_dew / count
    avg_wind = total_wind / count

    # ---------------------------------------------------------
    # Display Results
    # ---------------------------------------------------------

    print("\n------ WEATHER DATA ANALYSIS ------")
    print("Total Records :", count)

    print("\nAverage Temperature :", round(avg_temp, 2))
    print("Average Dew Point   :", round(avg_dew, 2))
    print("Average Wind Speed  :", round(avg_wind, 2))

    # ---------------------------------------------------------
    # LINE CHART VISUALIZATION
    # ---------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(dates, temperatures, marker='o', label='Temperature')
    plt.plot(dates, dew_points, marker='o', label='Dew Point')
    plt.plot(dates, wind_speeds, marker='o', label='Wind Speed')

    plt.title("Weather Data Analysis")
    plt.xlabel("Date")
    plt.ylabel("Values")
    plt.xticks(rotation=45)

    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# Run Program
# ---------------------------------------------------------

calculate_averages("sample_weather.txt")

# Date Temp DewPoint WindSpeed
# 2023-01-01 25 18 12
# 2023-01-02 27 19 10
# 2023-01-03 26 17 14
# 2023-01-04 28 20 11
# 2023-01-05 24 16 13
# 2023-01-06 29 21 15
# 2023-01-07 30 22 9