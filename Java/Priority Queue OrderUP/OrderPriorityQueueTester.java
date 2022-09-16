//////////////// FILE HEADER (INCLUDE IN EVERY FILE) //////////////////////////
//
// Title:    P10 Order UP v2
// Course:   CS 300 Spring 2021
//
// Author:   Drew Levin
// Email:    dslevin2@wisc.edu
// Lecturer: Mouna Kacem
//
///////////////////////// ALWAYS CREDIT OUTSIDE HELP //////////////////////////
//
// Persons:         Drew Levin
// Online Sources:  Documentation
//
///////////////////////////////////////////////////////////////////////////////

import java.util.Arrays;
import java.util.NoSuchElementException;

/**
 * This class checks the correctness of the implementation of the methods defined in the class
 * OrderPriorityQueue.
 * <p>
 * You MAY add additional public static boolean methods to this class if you like, and any private
 * static helper methods you need.
 */
public class OrderPriorityQueueTester {

    /**
     * Checks the correctness of the isEmpty method of OrderPriorityQueue.
     * <p>
     * You should, at least:
     * (1) create a new OrderPriorityQueue and verify that it is empty
     * (2) add a new Order to the queue and verify that it is NOT empty
     * (3) remove that Order from the queue and verify that it is empty again
     *
     * @return true if and only if ALL tests pass
     */
    public static boolean testIsEmpty() {
        Order.resetIDGenerator();

        // TODO implement scenario 1, then go write the constructor and isEmpty methods in your
        // OrderPriorityQueue class so that they pass the tests

        OrderPriorityQueue myQue = new OrderPriorityQueue(10);

        if (!myQue.isEmpty()) {
            return false;
        }

        // TODO implement scenario 2, then go write enough of insert() to pass the tests
        Order order1 = new Order("Pizza", 20);
        myQue.insert(order1);

        if (myQue.isEmpty()) {
            return false;
        }

        // TODO implement scenario 3, then go write enough of remove() to pass the tests
        myQue.removeBest();

        if (!myQue.isEmpty()) {
            return false;
        }

        return true;
    }

    /**
     * Checks the correctness of the insert method of OrderPriorityQueue.
     * <p>
     * You should, at least:
     * (1) create a new OrderPriorityQueue and add a single order with a large prepTime to it
     * (2) use the OrderPriorityQueue toString method to verify that the queue's internal structure
     * is a valid heap
     * (3) add at least three more orders with DECREASING prepTimes to the queue and repeat step 2.
     *
     * @return true if and only if ALL tests pass
     */
    public static boolean testInsertBasic() {
        Order.resetIDGenerator();

        // TODO implement this method, then go write the insert method in your
        // OrderPriorityQueue class so that it passes this test method
        OrderPriorityQueue myQue1 = new OrderPriorityQueue(10);
        Order order11 = new Order("Pizza", 20);
        Order order12 = new Order("Mac and Cheese", 15);
        Order order13 = new Order("Steak", 10);

        myQue1.insert(order11);

        if (!myQue1.toString().equals("1001(20)")) {
            return false;
        }


        myQue1.insert(order12);
        myQue1.insert(order13);

        if (!myQue1.toString().equals("1001(20), 1002(15), 1003(10)")) {
            return false;
        }


        return true; // included to prevent compiler errors
    }

    /**
     * Checks the correctness of the insert method of OrderPriorityQueue.
     * <p>
     * You should, at least:
     * (1) create an array of at least four Orders that represents a valid heap
     * (2) add a fifth order at the next available index that is NOT in a valid heap position
     * (3) pass this array to OrderPriorityQueue.percolateUp()
     * (4) verify that the resulting array is a valid heap
     *
     * @return true if and only if ALL tests pass
     */
    public static boolean testPercolateUp() {
        Order.resetIDGenerator();

        // TODO implement this method, then go write the percolateUp() method in your
        // OrderPriorityQueue class so that it passes this test method
        Order[] myQue2;
        Order order11 = new Order("Pizza", 20);
        Order order22 = new Order("Mac and Cheese", 15);
        Order order33 = new Order("Steak", 10);
        Order order44 = new Order("Chicken", 5);
        Order order55 = new Order("Sandwitch", 30);
        myQue2 = new Order[]{order11, order22, order33, order44, order55};
        OrderPriorityQueue.percolateUp(myQue2, 4);


        if (myQue2[0].getPrepTime() != 30) {
            return false;
        }
        if (myQue2[1].getPrepTime() != 20) {
            return false;
        }


        return true;
    }

    /**
     * Checks the correctness of the insert method of OrderPriorityQueue.
     * <p>
     * You should, at least:
     * (1) create a new OrderPriorityQueue with at least 6 orders of varying prepTimes, adding them
     * to the queue OUT of order
     * (2) use the OrderPriorityQueue toString method to verify that the queue's internal structure
     * is a valid heap
     *
     * @return true if and only if ALL tests pass
     */
    public static boolean testInsertAdvanced() {
        Order.resetIDGenerator();

        OrderPriorityQueue myQue3 = new OrderPriorityQueue(10);
        Order order111 = new Order("Pizza", 20);
        Order order222 = new Order("Mac and Cheese", 15);
        Order order333 = new Order("Steak", 10);
        Order order444 = new Order("Chicken", 5);
        Order order555 = new Order("Sandy", 30);
        Order order666 = new Order("Eggs", 9);


        // TODO implement this method, then go modify the insert method in your
        // OrderPriorityQueue class so that it passes this test method too
        myQue3.insert(order333);
        myQue3.insert(order111);
        myQue3.insert(order666);
        myQue3.insert(order444);
        myQue3.insert(order222);
        myQue3.insert(order555);


        if (!myQue3.toString().equals("1005(30), 1002(15), 1001(20), 1004(5), 1003(10), 1006(9)")) {
            return false;
        }

        return true; // included to prevent compiler errors
    }

    /**
     * Checks the correctness of the insert method of OrderPriorityQueue.
     * <p>
     * You should, at least:
     * (1) create an array of at least five Orders where the Order at index 0 is NOT in valid heap
     * position
     * (2) pass this array to OrderPriorityQueue.percolateDown()
     * (3) verify that the resulting array is a valid heap
     *
     * @return true if and only if ALL tests pass
     */
    public static boolean testPercolateDown() {
        Order.resetIDGenerator();


        // TODO implement this method, then go write the percolateUp() method in your
        // OrderPriorityQueue class so that it passes this test method

        Order[] myQue4;
        Order[] myQue44;
        Order[] myQue444;
        Order[] myQue123;
        Order order1111 = new Order("Pizza", 20);
        Order order2222 = new Order("Mac and Cheese", 15);
        Order order3333 = new Order("Steak", 10);
        Order order4444 = new Order("Chicken", 5);
        Order order5555 = new Order("Sandwitch", 30);
        Order order6666 = new Order("Eggs", 9);

        myQue4 = new Order[]{order2222, order5555, order1111, order4444, order6666};
        myQue444 = new Order[]{order2222, order5555, order1111, order4444, order6666};
        myQue44 = new Order[]{order5555, order2222, order1111, order4444, order6666};
        OrderPriorityQueue.percolateDown(myQue4, 0, myQue4.length);

        myQue123 = new Order[]{order6666, order5555, order3333, order2222, order1111, order4444};
        OrderPriorityQueue.percolateDown(myQue123, 0, myQue123.length);


        if (myQue123[0].getPrepTime() != 30) {
            return false;
        }
        if (myQue123[1].getPrepTime() != 20) {
            return false;
        }

        if (myQue123[2].getPrepTime() != 10) {
            return false;
        }
        if (myQue123[3].getPrepTime() != 15) {
            return false;
        }

        if (myQue123[4].getPrepTime() != 9) {
            return false;
        }
        if (myQue123[5].getPrepTime() != 5) {
            return false;
        }

        if (myQue4[0].getPrepTime() != 30) {
            return false;
        }
        if (myQue4[1].getPrepTime() != 15) {
            return false;
        }

        if (!Arrays.equals(myQue44, myQue4)) {
            return false;
        }

        if (Arrays.equals(myQue444, myQue4)) {
            return false;
        }


        return true; // included to prevent compiler errors
    }

    /**
     * Checks the correctness of the removeBest and peekBest methods of OrderPriorityQueue.
     * <p>
     * You should, at least:
     * (1) create a new OrderPriorityQueue with at least 6 orders of varying prepTimes, adding them
     * to the queue in whatever order you like
     * (2) remove all but one of the orders, verifying that each order has a SHORTER prepTime than
     * the previously-removed order
     * (3) peek to see that the only order left is the one with the SHORTEST prepTime
     * (4) check isEmpty to verify that the queue has NOT been emptied
     * (5) remove the last order and check isEmpty to verify that the queue HAS been emptied
     *
     * @return true if and only if ALL tests pass
     */
    public static boolean testPeekRemove() {
        try {
            Order.resetIDGenerator();

            // TODO implement this method, then go write the peek and dequeue methods in your
            // OrderPriorityQueue class so that they pass your tests
            OrderPriorityQueue myQue5 = new OrderPriorityQueue(10);
            Order order11111 = new Order("Pizza", 20);
            Order order22222 = new Order("Mac and Cheese", 15);
            Order order33333 = new Order("Steak", 10);
            Order order44444 = new Order("Chicken", 5);
            Order order55555 = new Order("Sandy", 30);
            Order order66666 = new Order("Eggs", 9);
            myQue5.insert(order33333);
            myQue5.insert(order11111);
            myQue5.insert(order66666);
            myQue5.insert(order22222);
            myQue5.insert(order55555);
            myQue5.insert(order44444);


            myQue5.removeBest();


            if (!myQue5.toString().equals("1001(20), 1002(15), 1006(9), 1003(10), 1004(5)")) {
                return false;
            }

            myQue5.removeBest();
            if (!myQue5.toString().equals("1002(15), 1003(10), 1006(9), 1004(5)")) {
                return false;
            }

            myQue5.removeBest();
            if (!myQue5.toString().equals("1003(10), 1004(5), 1006(9)")) {
                return false;
            }

            myQue5.removeBest();
            if (!myQue5.toString().equals("1006(9), 1004(5)")) {
                return false;
            }


            myQue5.removeBest();
            if (!myQue5.toString().equals("1004(5)")) {
                return false;
            }

            Order savedString = myQue5.peekBest();

            if (!myQue5.peekBest().equals(savedString)) {
                return false;
            }
            if (myQue5.isEmpty()) {
                return false;
            }
            myQue5.removeBest();

            if (!myQue5.isEmpty()) {
                return false;
            }

            return true;
        } catch (Exception e) {
            System.out.println(e);
        }
        return false;
    }

    /**
     * Checks the correctness of the removeBest and peekBest methods, as well as the constructor of
     * the OrderPriorityQueue class for erroneous inputs and/or states
     * <p>
     * You should, at least:
     * (1) create a new OrderPriorityQueue with an invalid capacity argument, and verify that the
     * correct exception is thrown
     * (2) call peekBest() on an OrderPriorityQueue with an invalid state for peeking, and verify that
     * the correct exception is thrown
     * (3) call removeBest() on an OrderPriorityQueue with an invalid state for removing, and verify
     * that the correct exception is thrown
     *
     * @return true if and only if ALL tests pass
     */
    public static boolean testErrors() {
        Order.resetIDGenerator();

        // TODO implement this method, then go modify the relevant methods in your
        // OrderPriorityQueue class so that they pass your tests
        for (int i = 0; i <= 3; i++) {
            if (i == 0) {
                try {
                    OrderPriorityQueue myQue6 = new OrderPriorityQueue(-2);
                } catch (IllegalArgumentException e) {
                    continue;
                } catch (Exception e) {
                    return false;
                }
            } else if (i == 1) {
                try {
                    OrderPriorityQueue myQue7 = new OrderPriorityQueue(2);
                    myQue7.peekBest();
                } catch (NoSuchElementException e) {
                    continue;
                } catch (Exception e) {
                    return false;
                }

            } else if (i == 2) {
                try {
                    OrderPriorityQueue myQue8 = new OrderPriorityQueue(2);
                    myQue8.removeBest();
                } catch (NoSuchElementException e) {
                    continue;
                } catch (Exception e) {
                    return false;
                }
            }
        }


        return true; // included to prevent compiler errors
    }

    /**
     * Calls the test methods individually and displays their output
     *
     * @param args
     */
    public static void main(String[] args) {
        try {
            System.out.println("isEmpty: " + testIsEmpty());
            System.out.println("insert basic: " + testInsertBasic());
            System.out.println("percolate UP: " + testPercolateUp());
            System.out.println("insert advanced: " + testInsertAdvanced());
            System.out.println("percolate DOWN: " + testPercolateDown());
            System.out.println("peek/remove valid: " + testPeekRemove());
            System.out.println("error: " + testErrors());
        } catch (Exception e) {
            System.out.print(e);
        }

    }

}