
import java.io.*;
import java.util.ArrayList;

/**
 * Dataset Class
 *
 * Load and save training/testing sets
 */
public class Dataset {
    // class variable
    // For training set
    private final ArrayList<Double[]> trainingFeatures;
    private final ArrayList<Integer> trainingLabels;
    // For testing set
    private final ArrayList<Double[]> testingFeatures;
    // Dataset characteristics
    private final String csvFileLocation; // location of dataset csv file
    private final String testSetLocation; // location of test set
    private final int label0;
    private final int label1; // the class that wil be categorized as label 0 and 1
    private final int featureDimension; // dimension of features for dataset

    /**
     * Constructor for the Dataset Instance
     *
     * @param csvFileLocation Require file location for the training set
     * @param testSetLocation Require test set location
     * @param label0 The class that will be categorized as label 0
     * @param label1 The class that will be categorized as label 1
     * @param featureDimension The feature's dimension
     */
    Dataset(String csvFileLocation, String testSetLocation, int label0, int label1, int featureDimension) {
        // Assign dataset characteristics
        this.csvFileLocation = csvFileLocation;
        this.testSetLocation = testSetLocation;
        this.label0 = label0;
        this.label1 = label1;
        this.featureDimension = featureDimension;

        // Make new empty ArrayLists to store dataset
        trainingFeatures = new ArrayList<>();
        trainingLabels = new ArrayList<>();
        testingFeatures = new ArrayList<>();
    }

    /**
     * Method to load training features and labels to the instance
     */
    public void loadTrainingSet() throws IOException{
        BufferedReader reader = new BufferedReader(new FileReader(csvFileLocation));
        String datapoint = reader.readLine(); // read Line one by one (one of each line indicating one datapoint)
        while(datapoint != null) {
            String[] labelAndFeatureEntries = datapoint.split(","); // csv file is split by comma
            int classLabel = Integer.parseInt(labelAndFeatureEntries[0]); // get classLabel for the datapoint

            // only interested in the cases with class label with label0 or label1
            if(classLabel == label0) {
                // add training instance
                trainingLabels.add(0);
                trainingFeatures.add(loadTrainingFeatures(labelAndFeatureEntries));
            } else if(classLabel == label1) {
                // add training instance
                trainingLabels.add(1);
                trainingFeatures.add(loadTrainingFeatures(labelAndFeatureEntries));
            }

            // get the next line
            datapoint = reader.readLine();
        }
    }

    /**
     * Helper method of loadTrianingSet.
     * This method converts String features to proper double value
     *
     * @param labelAndFeatureEntries String array containing information of one datapoint read from CSV file
     * @return Double array containing training features
     */
    private Double[] loadTrainingFeatures(String[] labelAndFeatureEntries) {
        // Create new Double array
        Double[] features = new Double[featureDimension];

        // convert String to proper Double
        for(int i = 1; i < featureDimension + 1; i++) {
            double pixelIntensity = Integer.parseInt(labelAndFeatureEntries[i]);
            pixelIntensity = pixelIntensity / 255; // pixel Intensity should be [0, 1]
            features[i - 1] = pixelIntensity;
        }

        return features;
    }

    /**
     * Method to load testing features to the instance
     */
    public void loadTestingSet() throws IOException{
        BufferedReader reader = new BufferedReader(new FileReader(testSetLocation));
        String datapoint = reader.readLine(); // read Line one by one (one of each line indicating one datapoint)
        while(datapoint != null) {
            String[] featureEntries = datapoint.split(","); // csv file is split by comma
            testingFeatures.add(loadTestingFeatures(featureEntries)); // add testing instance

            // get the next line
            datapoint = reader.readLine();
        }
    }

    /**
     * Helper method of loadTestingSet.
     * This method converts String features to proper double value
     *
     * @param featureEntries String array containing information of one datapoint read from CSV file
     * @return Double array containing training features
     */
    private Double[] loadTestingFeatures(String[] featureEntries) {
        // Create new Double array
        Double[] features = new Double[featureDimension];

        // convert String to proper Double
        for(int i = 0; i < featureDimension; i++) {
            double pixelIntensity = Integer.parseInt(featureEntries[i]);
            pixelIntensity = pixelIntensity / 255; // pixel Intensity should be [0, 1]
            features[i] = pixelIntensity;
        }

        return features;
    }

    /**
     * Accessor of trainingFeatures
     *
     * @return ArrayList of Double array containing trainingFeatures
     */
    public ArrayList<Double[]> getTrainingFeatures() {
        return trainingFeatures;
    }

    /**
     * Accessor of trainingLabels
     *
     * @return ArrayList of Integers containing trainingLabel
     */
    public ArrayList<Integer> getTrainingLabels() {
        return trainingLabels;
    }

    /**
     * Accessor of testingFeatures
     *
     * @return ArrayList of Double array containing testingFeatures
     */
    public ArrayList<Double[]> getTestingFeatures() {
        return testingFeatures;
    }
}
