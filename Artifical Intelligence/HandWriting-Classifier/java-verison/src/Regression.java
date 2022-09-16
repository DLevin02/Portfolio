// Written by: Drew Levin


import java.io.*;
import java.text.DecimalFormat;
import java.util.*;
import java.lang.Math;


public class Regression {

    private static final String COMMA_DELIMITER = ",";
    private static final String PATH_TO_TRAIN = "./src/mnist_train.csv";
    private static final String NEW_TEST = "./src/test.txt";
    private static final String outputPath = "./src/output.txt";
    private static final int MAX_EPOCHS = 10;
    static double alpha = 0.1;


    static String first_digit = "9";
    static String second_digit = "3";
    static Random rng = new Random();

    static double[][] wi = new double[28][785];
    static double[] wh = new double[29];

    public static double diff_sigg(double activation) {
        return activation * (1 - activation);
    }
    public static double sig(double sum, double bias) {
        return 1.0 / (1.0 + Math.exp(-1 * (sum + bias)));
    }

    public static double[][] parseRecords(String file_path) throws IOException {
        double[][] records = new double[20000][786];
        try (BufferedReader br = new BufferedReader(new FileReader(file_path))) {
            String line;
            int k = 0;
            while ((line = br.readLine()) != null) {

                String[] string_values = line.split(COMMA_DELIMITER);
                if (!string_values[0].equals(first_digit) && !string_values[0].contentEquals(second_digit)) continue;
                if (first_digit.equals(string_values[0])) records[k][0] = 0.0; // label 0
                else records[k][0] = 1.0; // label 1
                for (int i = 1; i < string_values.length; i++) {
                    records[k][i] = Double.parseDouble(string_values[i]) / 255.0; // features
                }
                records[k][785] = 1.0;

                k += 1;
            }

            double[][] res = new double[k][786];
            for (int i = 0; i < k; i++) {
                System.arraycopy(records[i], 0, res[i], 0, 786);
            }
            return res;
        }

    }

    public static double[][] NewTest(String file_path) throws IOException {
        double[][] records = new double[20000][785];
        try (BufferedReader br = new BufferedReader(new FileReader(file_path))) {
            String line;
            int k = 0;
            while ((line = br.readLine()) != null) {

                String[] string_values = line.split(COMMA_DELIMITER);
                for (int i = 0; i < string_values.length; i++) {
                    records[k][i] = Double.parseDouble(string_values[i]) / 255.0; // features
                }
                records[k][784] = 1.0;

                k += 1;
            }

            double[][] res = new double[k][785];
            for (int i = 0; i < k; i++) {
                System.arraycopy(records[i], 0, res[i], 0, 785);
            }
            return res;
        }

    }


    public static void main(String[] args) {
        startFile(outputPath);

        try (FileWriter file = new FileWriter(outputPath)) {
            file.write("output file here\n");

            double[][] train = parseRecords(PATH_TO_TRAIN);
            double[][] new_test = NewTest(NEW_TEST);

            int num_train = train.length;
            int num_test = new_test.length;

            for (int i = 0; i < wi.length; i++) {
                for (int j = 0; j < wi[0].length; j++) {
                    wi[i][j] = 2 * rng.nextDouble() - 1;
                }
            }
            for (int i = 0; i < wh.length; i++) {
                wh[i] = 2 * rng.nextDouble() - 1;
            }


            for (int epoch = 1; epoch <= MAX_EPOCHS; epoch++) {
                double[] out_o = new double[num_train];
                double[][] out_h = new double[num_train][29];

                for (int index = 0; index < num_train; index++) {
                    double[] row = train[index];
                    double label = row[0];
                    for (int i = 0; i < num_train; ++i) {
                        out_h[i][28] = 1.0;
                    }

                    for (int i = 0; i < 28; i++) {
                        double s = 0.0;
                        for (int j = 0; j < 785; j++) {
                            s += wi[i][j] * row[j + 1];
                        }
                        out_h[index][i] = sig(s, out_h[i][28]);
                    }

                    double num = 0.0;
                    out_o[28] = rng.nextDouble();
                    for (int i = 0; i < 29; i++) {
                        num += out_h[index][i] * wh[i];
                    }
                    out_o[index] = sig(num, out_o[28]);

                    double[] delta = new double[29];
                    for (int i = 0; i < 29; i++) {
                        delta[i] = diff_sigg(out_h[index][i]) * wh[i] * (label - out_o[index]);
                    }

                    for (int i = 0; i < 28; i++) {
                        for (int j = 0; j < 785; j++) {
                            wi[i][j] += alpha * delta[i] * row[j + 1];
                        }
                    }

                    for (int i = 0; i < 29; ++i) {
                        wh[i] += alpha * (label - out_o[index]) * out_h[index][i];
                    }

                }

                double error = 0;
                for (int ind = 0; ind < num_train; ind++) {
                    double[] row = train[ind];
                    error += -row[0] * Math.log(out_o[ind]) - (1 - row[0]) * Math.log(1 - out_o[ind]);
                }

                double perfect = 0.0;
                for (int ind = 0; ind < num_train; ind++) {
                    if ((train[ind][0] == 1.0 && out_o[ind] >= 0.5) || (train[ind][0] == 0.0 && out_o[ind] < 0.5))
                        perfect += 1.0;
                }
                double actual = perfect / num_train;

                file.write("Epoch: " + epoch + ", error: " + error + ", actual: " + actual + "\n");
                System.out.println("Epoch: " + epoch + ", error: " + error + ", actual: " + actual);
            }

            weights(file, wi, wh);

            double[][] test_ih = new double[new_test.length][28];
            double[] test_ho = new double[new_test.length];
            for (int index = 0; index < num_test; index++) {
                double[] r = new_test[index];

                for (int i = 0; i < 28; ++i) {
                    double s = 0.0;
                    for (int j = 0; j < 784; ++j) {
                        s += r[j] * wi[i][j];
                    }

                    test_ih[index][i] = sig(s, wi[i][784]);
                }

                double s = 0.0;
                for (int i = 0; i < 28; i++) {
                    s += test_ih[index][i] * wh[i];
                }

                test_ho[index] = sig(s, wh[28]);
            }

            printTestSet(file, test_ho, new_test);

        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
    }

    private static void printTestSet(FileWriter file, double[] test_ho, double[][] test) throws IOException {
        DecimalFormat df = new DecimalFormat("0.00");
        file.write("\nSecondLayer\n");

        for (int i = 0; i < test_ho.length; i++) {
            file.write(df.format(test_ho[i]));

            if (i < test_ho.length - 1) {
                file.write(",");
            } else {
                file.write("\n");
            }
        }

        DecimalFormat rounded = new DecimalFormat("0");
        file.write("\nTest set, predicted values:\n");
        for (int i = 0; i < test_ho.length; i++) {
            file.write(rounded.format(test_ho[i]));

            if (i < test_ho.length - 1) {
                file.write(",");
            } else {
                file.write("\n");
            }
        }

        double minD = 100;
        int minI = -1;
        for (int i = 0; i < test_ho.length; i++) {
            if (Math.abs(test_ho[i] - 0.5) < minD) {
                minD = test_ho[i];
                minI = i;
            }
        }
        file.write("Test image index closest to 0.5: " + minI + "\n");

        // print feature vector of test image closest to 0.5
        DecimalFormat twoPlaces = new DecimalFormat("0.00");
        file.write("\nFeature vector of test image closest to 0.5: \n");

        for (int i = 0; i < test[minI].length; i++) {
            file.write(twoPlaces.format(test[minI][i]));

            if (i < test[minI].length - 1) {
                file.write(",");
            } else {
                file.write("\n");
            }
        }
    }

    private static void weights(FileWriter file, double[][] wi, double[] wh) throws IOException {
        DecimalFormat deciimal = new DecimalFormat("0.0000");

        file.write("\nFirst layer\n");
        for (int i = 0; i < 785; i++) {
            for (int j = 0; j < 28; j++) {
                file.write(deciimal.format(wi[j][i]));
                if (j < wi.length - 1) {
                    file.write(",");
                } else {
                    file.write("\n");
                }
            }
        }

        file.write("\nFirst layer\n");
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 785; j++) {
                file.write(deciimal.format(wi[i][j]));
                if (j < 785 - 1) {
                    file.write(",");
                } else {
                    file.write("\n");
                }
            }
        }
        file.write("\nSecond layer");
        for (int i = 0; i < wh.length; i++) {
            file.write(deciimal.format(wh[i]));
            if (i < wh.length - 1) {
                file.write(",");
            } else {
                file.write("\n");
            }
        }
    }

    private static void startFile(String outputPath) {
        try {
            File newfile = new File(outputPath);
            if (newfile.createNewFile()) {
                System.out.println("File Created: " + newfile.getName());
            } else {
                System.out.println("File EXIST.");
            }
        } catch (IOException e) {
            System.out.println("ERROR.");
            e.printStackTrace();
        }
    }

}