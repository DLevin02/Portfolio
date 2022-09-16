// Written by: Young Wu
// Attribution: Hugh Liu's CS540 P1 Solution 2020

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;
import java.text.DecimalFormat;
import java.util.List;


public class RegressionP1 {
    // TODO: change hyper-parameters HERE, like iterations, learning_rate, etc.
    private static final String COMMA_DELIMITER = ",";
    private static final String PATH_TO_TRAIN = "./src/mnist_train.csv";
    private static final String NEW_TEST = "./src/test.txt";
    private static final int MAX_EPOCHS = 10;
    private static final double ALPHA = 0.1;
    private static final String[] LABELS = new String[] { "9", "3" };
    private static final Random rng = new Random();

    public static double[][] parseRecords(String file_path) throws IOException {
        double[][] records = new double[20000][786];
        try (BufferedReader br = new BufferedReader(new FileReader(file_path))) {
            String line;
            int k = 0;
            while ((line = br.readLine()) != null) {
                String[] string_values = line.split(COMMA_DELIMITER);
                if (!string_values[0].equals(LABELS[0]) && !string_values[0].contentEquals(LABELS[1]))
                    continue;
                if (LABELS[0].equals(string_values[0]))
                    records[k][0] = 0.0; // label 0
                else
                    records[k][0] = 1.0; // label 1
                for (int i = 1; i < string_values.length; i++)
                    records[k][i] = Double.parseDouble(string_values[i]) / 255.0; // features
                records[k][785] = 1.0;
                k += 1;
            }
            double[][] res = new double[k][786];
            for (int i = 0; i < k; i++)
                System.arraycopy(records[i], 0, res[i], 0, 786);
            return res;
        }
    }

    private static void printBiasAndWeight(Double b, Double[] w) {
        DecimalFormat df = new DecimalFormat("0.0000");

        System.out.println("Bias and weight: ");
        System.out.printf(b + ",");
        for (int i = 0; i < w.length; i++) {
            System.out.printf(df.format(w[i]));
            if (i < w.length - 1) {
                System.out.printf(",");
            } else {
                System.out.println();
            }
        }
    }

    private static void printFeatureVector(List<List<Double>> test_records, int sampleToPrint) {
        Double[] ary = test_records.get(sampleToPrint).toArray(new Double[0]);
        DecimalFormat df = new DecimalFormat("0.00");

        System.out.println("Feature vector: ");
        for (int i = 1; i < ary.length; i++) {
            System.out.printf(df.format(ary[i]));
            if (i < ary.length - 1) {
                System.out.printf(",");
            } else {
                System.out.println();
            }
        }
    }


    public static double[][] NewTest(String file_path) throws IOException {
        double[][] records = new double[20000][785];
        try (BufferedReader br = new BufferedReader(new FileReader(file_path))) {
            String line;
            int k = 0;
            while ((line = br.readLine()) != null) {
                String[] string_values = line.split(COMMA_DELIMITER);
                for (int i = 0; i < string_values.length; i++)
                    records[k][i] = Double.parseDouble(string_values[i]) / 255.0; // features
                records[k][784] = 1.0;
                k += 1;
            }
            double[][] res = new double[k][785];
            for (int i = 0; i < k; i++)
                System.arraycopy(records[i], 0, res[i], 0, 785);
            return res;
        }
    }

    public static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public static void main(String[] args) throws IOException {
        double[][] train = parseRecords(PATH_TO_TRAIN);
        int m = 784; // number of inputs
        double[] w = new double[m];
        double b = 2 * rng.nextDouble() - 1;
        for (int i = 0; i < w.length; i++)
            w[i] = 2 * rng.nextDouble() - 1; // initialize weights
        int num_train = train.length;
        double loss_prev = 0.0;
        for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
            // calculate a_i array
            double[] a = new double[num_train];
            for (int ind = 0; ind < num_train; ind++) {
                double s = 0;
                for (int i = 0; i < w.length; i++)
                    s += w[i] * train[ind][i + 1];
                a[ind] = sigmoid(s + b);
            }
            // update weights
            for (int j = 0; j < w.length; j++) {
                double dw = 0.0;
                for (int i = 0; i < num_train; i++)
                    dw += (a[i] - train[i][0]) * train[i][j + 1];
                w[j] -= ALPHA * dw;
            }
            // update bias
            double db = 0;
            for (int i = 0; i < num_train; i++)
                db += (a[i] - train[i][0]);
            b -= ALPHA * db;
            // calculate loss
            double loss = 0.0;
            for (int i = 0; i < num_train; i++) {
                if (train[i][0] == 0.0) {
                    if (a[i] > 0.9999)
                        loss += 100.0; // something large
                    else
                        loss -= Math.log(1 - a[i]);
                } else if (train[i][0] == 1.0) {
                    if (a[i] < 0.0001)
                        loss += 100.0;
                    else
                        loss -= Math.log(a[i]);
                }
            }
            double loss_reduction = loss_prev - loss;
            loss_prev = loss;
            // count correct
            double correct = 0.0;
            for (int ind = 0; ind < num_train; ind++) {
                if ((train[ind][0] == 1.0 && a[ind] >= 0.5) || (train[ind][0] == 0.0 && a[ind] < 0.5))
                    correct += 1.0;
            }
            double acc = correct / num_train;
            System.out.println("epoch = " + epoch + ", loss = " + loss + ", loss reduction = " + loss_reduction
                    + ", correctly classified = " + acc);
        }
        double[][] test = NewTest(NEW_TEST);
    }


}
