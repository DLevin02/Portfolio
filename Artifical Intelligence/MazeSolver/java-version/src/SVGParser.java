import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.parsers.DocumentBuilder;

import org.w3c.dom.Document;
import org.xml.sax.SAXException;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

public class SVGParser {

    private static String filePathName;
    private static int svgWidth;
    private static int svgHeight;
    private static int[][] svg;
    private static int mazeWidth;
    private static int mazeHeight;
    private static Map<String, ArrayList<String>> maze = new HashMap<>();

    public SVGParser(String filePathName, int width, int height) {
        SVGParser.filePathName = filePathName;
        SVGParser.mazeWidth = width;
        SVGParser.mazeHeight = height;
    }

    /*
     * This method parses svg file which has a structure of xml document.
     * It generates a 2D array for svg values where 1's represent walls, 0's are roads.
     *
     */
    public void parseXML() throws ParserConfigurationException, SAXException, IOException {

        System.out.println("Started Parsing XML Document.");

        DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
        DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
        Document doc = dBuilder.parse(new File(filePathName));

        doc.getDocumentElement().normalize();

        Element svgNode = (Element) doc.getDocumentElement();
        svgWidth = Integer.valueOf(svgNode.getAttribute("width").toString());
        svgHeight = Integer.valueOf(svgNode.getAttribute("height").toString());

        // initialize svg matrix, should be of the size svgWidth x svgHeight
        svg = new int[svgHeight][svgWidth];

        // get all elements with the line tag
        NodeList nList = doc.getElementsByTagName("line");

        for (int temp = 0; temp < nList.getLength(); temp++) {
            Node nNode = nList.item(temp);

            // get x1, y1, x2, y2 values for each line from xml file
            int x1 = Integer.valueOf(((Element) nNode).getAttribute("x1").toString()) - 1;
            int y1 = Integer.valueOf(((Element) nNode).getAttribute("y1").toString()) - 1;
            int x2 = Integer.valueOf(((Element) nNode).getAttribute("x2").toString()) - 1;
            int y2 = Integer.valueOf(((Element) nNode).getAttribute("y2").toString()) - 1;

            // update svg matrix according to the line coordinates, i.e. walls of the maze
            for (int i = x1; i < x2 + 1; i++) {
                for (int j = y1; j < y2 + 1; j++) {
                    svg[j][i] = 1;
                }
            }
        }
        System.out.println("Finished Parsing XML Document.");
    }

    /*
     * This method finds neighbors for every cell in the maze and stores
     * those values in the hash map. It makes use of the svg 2D array to identify
     * where the walls are in the maze.
     * Key - a string with x and y values being concatenated
     * Value - list of neighbors (letters U,D,L,R)
     *
     */
    public Cell[] generateMaze() {

        // ignoring some 0 values used as padding in the svg
        int delta_width = (svgWidth - 3) / mazeWidth;
        int delta_height = (svgHeight - 3) / mazeHeight;

        // go through every cell in the maze and identify neighbors or walls around it
        for (int row = 0; row < mazeHeight; row++) {
            for (int col = 0; col < mazeWidth; col++) {

                String cellCoordinate = String.valueOf(row + "+" + col);

                /*
                 * The list neighbors stores letters "U", "D", "L", "R" referring to
                 * top, bottom, left, right neighbors of the cell.
                 * If there is a letter, then this cell has a neighbor on that side of it.
                 * List size is equal to the number of neighbors for that cell.
                 */
                ArrayList<String> neighbors = new ArrayList<>();

                // check top
                if (svg[row * delta_height + 1][col * delta_width + 1 + delta_width / 2] == 0) {
                    // there is no wall on the top
                    neighbors.add("U");
                }

                // check bottom
                if (svg[(row + 1) * delta_height + 1][col * delta_width + 1 + delta_width / 2] == 0) {
                    // there is no wall on the bottom
                    neighbors.add("D");
                }

                // check left
                if (svg[row * delta_height + 1 + delta_height / 2][col * delta_width + 1] == 0) {
                    // there is no wall on the left
                    neighbors.add("L");
                }

                // check right
                if (svg[row * delta_height + 1 + delta_height / 2][(col + 1) * delta_width + 1] == 0) {
                    // there is no wall on the right
                    neighbors.add("R");
                }
                // this map contains neighbors list for each coordinate of the maze
                maze.put(cellCoordinate, neighbors);
            }
        }

        int[] startCoord = findStartCoordinates();
        int[] finishCoord = findFinishCoordinates();

        // create Cell objects for every cell in the maze, initialize neighbors
        // and return start and finish cells in the array
        Cell[] startAndFinishCells = generateCellObjects(startCoord, finishCoord);

        return startAndFinishCells;
    }

    /*
     * This method returns the array with two values x and y for the start coordinate of the maze.
     */
    public int[] findStartCoordinates() {

        int[] startCoord = new int[2];
        startCoord[1] = 0; // y coordinate for start of the maze, assuming that it's at the top

        for (int i = 0; i < mazeWidth; i++) {
            ArrayList<String> neighbors = maze.get("0+" + String.valueOf(i));

            if (neighbors != null) {
                // find cell on the first row without wall on the top
                if (neighbors.contains("U")) {
                    startCoord[0] = i; // x coordinate for start on the maze
                }
            }

        }

        System.out.println("\nStart coordinates of the maze: x = " + startCoord[0] +
                " y = " + startCoord[1]);

        return startCoord;
    }

    /*
     * This method returns the array with two values x and y for the finish coordinate of the maze.
     */
    public int[] findFinishCoordinates() {

        int[] finishCoord = new int[2];
        finishCoord[1] = mazeHeight - 1; // y coordinate for finish of the maze, assuming that it's at the bottom

        for (int i = 0; i < mazeWidth; i++) {
            ArrayList<String> neighbors = maze.get(String.valueOf(mazeHeight - 1) + "+" + String.valueOf(i));

            if (neighbors != null) {
                // find cell on the last row without wall on the bottom
                if (neighbors.contains("D")) {
                    finishCoord[0] = i; // x coordinate for finish on the maze
                }
            }

        }

        System.out.println("Finish coordinates of the maze: x = " + finishCoord[0] +
                " y = " + finishCoord[1]);

        return finishCoord;
    }

    /*
     * This method is needed for initializing all Cell objects of our maze.
     * Also, it is responsible for linking the list of neighbors to every cell in the maze.
     *
     * It returns the array containing start and finish Cell objects that are enough
     * to run our search algorithms.
     */
    public Cell[] generateCellObjects(int[] startCoord, int[] finishCoord) {

        Cell start = null;
        Cell finish = null;

        // for all cells in the maze create object of type Cell
        HashMap<String, Cell> mapOfCells = new HashMap<>();
        for (int row = 0; row < mazeHeight; row++) {
            for (int col = 0; col < mazeWidth; col++) {
                Cell c;
                if (row == startCoord[1] && col == startCoord[0]) {
                    // Cell should be start
                    c = new Cell(col, row);
                    start = c;
                } else if (row == finishCoord[1] && col == finishCoord[0]) {
                    // cell is finish
                    c = new Cell(col, row);
                    finish = c;
                } else {
                    // other cells
                    c = new Cell(col, row);
                }
                mapOfCells.put(String.valueOf(c.yCoord + "+" + c.xCoord), c);
            }
        }

        // for every cell create a list of neighbors and link them to the cell object
        for (String key : maze.keySet()) {

            Cell c = mapOfCells.get(key);

            ArrayList<String> neighbors = maze.get(key);
            ArrayList<Cell> neighboringCells = new ArrayList<>();

            for (int ind = 0; ind < neighbors.size(); ind++) {

                Cell adjacentCell;
                int index = key.indexOf("+");
                int startXCoord = Integer.valueOf(key.substring(index + 1, key.length()));
                int startYCoord = Integer.valueOf(key.substring(0, index));

                if (neighbors.get(ind) == "U") {
                    if (startYCoord == 0) {
                        continue;
                    }
                    adjacentCell = mapOfCells.get(String.valueOf((startYCoord - 1) + "+" + startXCoord));
                } else if (neighbors.get(ind) == "D") {
                    if (startYCoord == mazeHeight - 1) {
                        continue;
                    }
                    adjacentCell = mapOfCells.get(String.valueOf((startYCoord + 1) + "+" + startXCoord));

                } else if (neighbors.get(ind) == "L") {
                    adjacentCell = mapOfCells.get(String.valueOf((startYCoord) + "+" + (startXCoord - 1)));

                } else {
                    adjacentCell = mapOfCells.get(String.valueOf((startYCoord) + "+" + (startXCoord + 1)));
                }

                neighboringCells.add(adjacentCell);
            }
            c.setNeighbors(neighboringCells);
        }
        Cell[] startAndFinishCells = new Cell[2];
        startAndFinishCells[0] = start;
        startAndFinishCells[1] = finish;
        return startAndFinishCells;
    }

    public void printSuccessorMatrix() {

        for (int row = 0; row < mazeHeight; row++) {
            System.out.println();
            for (int col = 0; col < mazeWidth; col++) {
                ArrayList<String> neighbors = maze.get(String.valueOf(row + "+" + col));

                StringBuilder sb = new StringBuilder();
                for (String succ : neighbors) {
                    if (succ.equals("U")) {
                        sb.append("U");
                    }
                    if (succ.equals("D")) {
                        sb.append("D");
                    }
                    if (succ.equals("L")) {
                        sb.append("L");
                    }
                    if (succ.equals("R")) {
                        sb.append("R");
                    }
                }
                System.out.print(sb.toString());
                if (col != mazeWidth - 1) {
                    System.out.print(",");
                }
            }
        }
        System.out.println();
    }

    /**
     * Print successor matrix to file
     */
    public void printToFileSuccessorMatrix(FileWriter writer) throws IOException {

        for (int row = 0; row < mazeHeight; row++) {
            for (int col = 0; col < mazeWidth; col++) {
                ArrayList<String> neighbors = maze.get(String.valueOf(row + "+" + col));

                StringBuilder sb = new StringBuilder();
                for (String succ : neighbors) {
                    if (succ.equals("U")) {
                        sb.append("U");
                    }
                    if (succ.equals("D")) {
                        sb.append("D");
                    }
                    if (succ.equals("L")) {
                        sb.append("L");
                    }
                    if (succ.equals("R")) {
                        sb.append("R");
                    }
                }
                writer.write(sb.toString());
                if (col != mazeWidth - 1) {
                    writer.write(",");
                }
            }
            writer.write("\n");
        }
    }

    /**
     * Print maze plot to file
     */
    public void printMaze(FileWriter writer) throws IOException {

        // ignoring some 0 values used as padding in the svg
        int delta_width = (svgWidth - 3) / mazeWidth;
        int delta_height = (svgHeight - 3) / mazeHeight;

        // go through every cell in the maze and identify neighbors or walls around it
        for (int row = 0; row < mazeHeight; row++) {
            for (int col = 0; col < mazeWidth; col++) { // check top of entire row
                writer.write("+"); // each cell begins with a +
                // check top
                if (svg[row * delta_height + 1][col * delta_width + 1 + delta_width / 2] == 0) { // no wall
                    writer.write("  ");
                } else { // wall
                    writer.write("--");
                }

                if (col == mazeWidth - 1) { // last col
                    writer.write("+"); // ends with a +
                    writer.write("\n");
                }
            }

            for (int col = 0; col < mazeWidth; col++) { // check left and right (middle of row)
                // check left
                if (svg[row * delta_height + 1 + delta_height / 2][col * delta_width + 1] == 0) { // no wall
                    writer.write(" ");
                } else { // wall
                    writer.write("|");
                }

                writer.write("  "); // cell middle is two spaces

                if (col == mazeWidth - 1) { // last col
                    if (svg[row * delta_height + 1 + delta_height / 2][(col + 1) * delta_width + 1] == 0) { // no wall
                        writer.write("");
                    } else { // wall
                        writer.write("|");
                    }
                    writer.write("\n");
                }
            }

            if (row == mazeHeight - 1) { // last row
                for (int col = 0; col < mazeWidth; col++) { // check bottom of entire row
                    writer.write("+"); // bottom row of each cell begins with a +
                    // check bot
                    if (svg[(row + 1) * delta_height + 1][col * delta_width + 1 + delta_width / 2] == 0) { // no wall
                        writer.write("  ");
                    } else { // wall
                        writer.write("--");
                    }

                    if (col == mazeWidth - 1) { // last col
                        writer.write("+"); // ends with a +
                    }
                }
            }
        }
    }

    /**
     * Print maze plot with solution to file
     */
    public void printMazeWithSolution(FileWriter writer, ArrayList<Integer[]> solutionPath) throws IOException {

        // ignoring some 0 values used as padding in the svg
        int delta_width = (svgWidth - 3) / mazeWidth;
        int delta_height = (svgHeight - 3) / mazeHeight;

        // go through every cell in the maze and identify neighbors or walls around it
        for (int row = 0; row < mazeHeight; row++) {
            for (int col = 0; col < mazeWidth; col++) { // check top of entire row
                boolean isSolutionPath = false;
                isSolutionPath = isSolutionPath(solutionPath, row, col, isSolutionPath);

                writer.write("+"); // each cell begins with a +
                // check top
                if (isSolutionPath) {
                    writer.write("@@");
                } else if (svg[row * delta_height + 1][col * delta_width + 1 + delta_width / 2] == 0) { // no wall
                    writer.write("  ");
                } else { // wall
                    writer.write("--");
                }

                if (col == mazeWidth - 1) { // last col
                    writer.write("+"); // ends with a +
                    writer.write("\n");
                }
            }

            for (int col = 0; col < mazeWidth; col++) { // check left and right (middle of row)
                boolean isSolutionPath = false;
                isSolutionPath = isSolutionPath(solutionPath, row, col, isSolutionPath);

                // check left
                if (svg[row * delta_height + 1 + delta_height / 2][col * delta_width + 1] == 0) { // no wall
                    writer.write(" ");
                } else { // wall
                    writer.write("|");
                }

                if (isSolutionPath) {
                    writer.write("@@"); // cell middle is two pounds
                } else {
                    writer.write("  "); // cell middle is two spaces
                }

                if (col == mazeWidth - 1) { // last col
                    if (svg[row * delta_height + 1 + delta_height / 2][(col + 1) * delta_width + 1] == 0) { // no wall
                        writer.write("");
                    } else { // wall
                        writer.write("|");
                    }
                    writer.write("\n");
                }
            }

            if (row == mazeHeight - 1) { // last row
                for (int col = 0; col < mazeWidth; col++) { // check bottom of entire row
                    boolean isSolutionPath = false;
                    isSolutionPath = isSolutionPath(solutionPath, row, col, isSolutionPath);

                    writer.write("+"); // bottom row of each cell begins with a +
                    // check bot
                    if (isSolutionPath) {
                        writer.write("@@");
                    } else if (svg[(row + 1) * delta_height + 1][col * delta_width + 1 + delta_width / 2] == 0) { // no wall
                        writer.write("  ");
                    } else { // wall
                        writer.write("--");
                    }

                    if (col == mazeWidth - 1) { // last col
                        writer.write("+"); // ends with a +
                    }
                }
            }
        }
    }

    private boolean isSolutionPath(ArrayList<Integer[]> solutionPath, int row, int col, boolean isSolutionPath) {
        for (Integer[] solutionCoordinates : solutionPath) {
            if (solutionCoordinates[1] == row && solutionCoordinates[0] == col) {
                isSolutionPath = true;
                break;
            }
        }
        return isSolutionPath;
    }

    public void printManhattanToFinish(FileWriter writer, Cell finish) throws IOException {
        for (int row = 0; row < mazeHeight; row++) {
            for (int col = 0; col < mazeWidth; col++) {
                int manhattanDistance = Math.abs(col - finish.xCoord) + Math.abs(row - finish.yCoord);
                writer.write(String.valueOf(manhattanDistance));
                if (col != mazeWidth - 1) {
                    writer.write(",");
                }
            }
            writer.write("\n");
        }
    }
}
