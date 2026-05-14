// ---------------------------------------------------------
// BIG DATA CONCEPT:
// Weather Data Analysis in Java
// ---------------------------------------------------------

import java.io.*;
import java.util.*;

public class WeatherAnalysis {

    public static void calculateAverages(String filename) {

        // Lists for storing data
        ArrayList<String> dates = new ArrayList<>();
        ArrayList<Double> temperatures = new ArrayList<>();
        ArrayList<Double> dewPoints = new ArrayList<>();
        ArrayList<Double> windSpeeds = new ArrayList<>();

        // Variables for averages
        double totalTemp = 0;
        double totalDew = 0;
        double totalWind = 0;
        int count = 0;

        try {

            // Open file
            BufferedReader br = new BufferedReader(new FileReader(filename));

            // Skip header
            br.readLine();

            String line;

            // Read each line
            while ((line = br.readLine()) != null) {

                String[] parts = line.split("\\s+");

                // Extract values
                String date = parts[0];
                double temp = Double.parseDouble(parts[1]);
                double dew = Double.parseDouble(parts[2]);
                double wind = Double.parseDouble(parts[3]);

                // Store data
                dates.add(date);
                temperatures.add(temp);
                dewPoints.add(dew);
                windSpeeds.add(wind);

                // Add totals
                totalTemp += temp;
                totalDew += dew;
                totalWind += wind;

                count++;
            }

            br.close();

            // Calculate averages
            double avgTemp = totalTemp / count;
            double avgDew = totalDew / count;
            double avgWind = totalWind / count;

            // ---------------------------------------------------------
            // Display Results
            // ---------------------------------------------------------

            System.out.println("\n------ WEATHER DATA ANALYSIS ------");
            System.out.println("Total Records : " + count);

            System.out.println("\nAverage Temperature : " + avgTemp);
            System.out.println("Average Dew Point   : " + avgDew);
            System.out.println("Average Wind Speed  : " + avgWind);

            // ---------------------------------------------------------
            // Display Stored Data
            // ---------------------------------------------------------

            System.out.println("\nDate\t\tTemp\tDew\tWind");

            for (int i = 0; i < dates.size(); i++) {

                System.out.println(
                    dates.get(i) + "\t" +
                    temperatures.get(i) + "\t" +
                    dewPoints.get(i) + "\t" +
                    windSpeeds.get(i)
                );
            }

        } catch (Exception e) {

            System.out.println("Error: " + e.getMessage());
        }
    }

    // ---------------------------------------------------------
    // Main Method
    // ---------------------------------------------------------

    public static void main(String[] args) {

        calculateAverages("sample_weather.txt");
    }
}
