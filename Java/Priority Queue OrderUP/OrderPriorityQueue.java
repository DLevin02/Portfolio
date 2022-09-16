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

import java.util.NoSuchElementException;

/**
 * A max-heap implementation of a priority queue for Orders, where the Order with the LONGEST prep
 * time is returned first instead of the strict first-in-first-out queue as in P08.
 * <p>
 * TODO: Make sure Order implements Comparable<Order> so that this class can implement the
 * PriorityQueueADT without error!
 */
public class OrderPriorityQueue implements PriorityQueueADT<Order> {

    // Data fields; do not modify
    private Order[] queueHeap;
    private int size;


    /**
     * Constructs a PriorityQueue for Orders with the given capacity
     *
     * @param capacity the initial capacity for the queue
     * @throws IllegalArgumentException if the given capacity is 0 or negative
     */
    public OrderPriorityQueue(int capacity) {
        // TODO throw IllegalArgumentException if capacity is invalid

        if (capacity <= 0) {
            throw new IllegalArgumentException();
        }

        // TODO initialize data fields appropriately
        size = 0;
        queueHeap = new Order[capacity];
    }

    /**
     * Inserts a new Order into the queue in the appropriate position using a heap's add logic.
     *
     * @param newOrder the Order to be added to the queue
     */
    @Override
    public void insert(Order newOrder) {
        // TODO If the queue is empty, insert the new order at the root of the heap
        if (isEmpty()) {
            queueHeap[0] = newOrder;
        }

        // TODO If the queue is FULL, create a new Order array of double the current heap's size,
        // copy all elements of the current heap over and update the queueHeap reference
        // -> HINT: use Arrays.copyOf(), copying arrays is not the point of this assignment


        // TODO add the newOrder to the end of the heap and percolate up to ensure a valid heap, where
        // the Order with the LONGEST prep time is at the root of the heap
        //resize if necessary
        if (this.size == this.queueHeap.length) {
            Order[] bigHeap = new Order[this.queueHeap.length * 2];
            for (int i = 0; i < this.queueHeap.length; i++) {
                bigHeap[i] = this.queueHeap[i];
            }
            this.queueHeap = bigHeap;
        }

        this.queueHeap[this.size] = newOrder;

        // Percolate up
        percolateUp(this.queueHeap, this.size);


        // Now that we're sure we managed to add this element
        ++size;
    }

    /**
     * A utility method to percolate Order values UP through the heap; see figure 13.3.1 in zyBooks
     * for a pseudocode algorithm.
     *
     * @param heap       an array containing the Order values to be percolated into a valid heap
     * @param orderIndex the index of the Order to be percolated up
     */
    protected static void percolateUp(Order[] heap, int orderIndex) {
        // TODO implement the max heap percolate up algorithm to modify the heap in-place
        while (orderIndex > 0) {
            if (heap[(orderIndex - 1) / 2].compareTo(heap[orderIndex]) < 0) {
                Order temp = heap[(orderIndex - 1) / 2];
                heap[(orderIndex - 1) / 2] = heap[orderIndex];
                heap[orderIndex] = temp;
                orderIndex = (orderIndex - 1) / 2;
                continue;
            } else {
                break;
            }
        }
    }

    /**
     * Return the Order with the longest prep time from the queue and adjust the queue accordingly
     *
     * @return the Order with the current longest prep time from the queue
     * @throws NoSuchElementException if the queue is empty
     */
    @Override
    public Order removeBest() {
        // TODO If the queue is empty, throw a NoSuchElementException
        if (this.isEmpty()) {
            throw new NoSuchElementException();
        }

        // TODO Remove the root Order of the heap and re-structure the heap to ensure that its ordering
        // is valid, then return the previous root

        Order result = queueHeap[0];

        // Replace the value
        // at the root with
        // the last leaf

        queueHeap[0] = queueHeap[size - 1];
        size = size - 1;

        // Shift down the replaced
        // element to maintain the
        // heap property
        percolateDown(this.queueHeap, 0, this.size);

        return result;
    }

    /**
     * A utility method to percolate Order values DOWN through the heap; see figure 13.3.2 in zyBooks
     * for a pseudocode algorithm.
     *
     * @param heap       an array containing the Order values to be percolated into a valid heap
     * @param orderIndex the index of the Order to be percolated down
     * @param size       the number of initialized elements in the heap
     */
    protected static void percolateDown(Order[] heap, int orderIndex, int size) {
        // TODO implement the max heap percolate down algorithm to modify the heap in-place

        int r = orderIndex;
        int c;
        int n = size;
        Order temp;

        c = 2 * r;

        while (c < n) {
            if (c < n - 1 && heap[c].compareTo(heap[c + 1]) < 0) {
                c++;
            }

            if (heap[r].compareTo(heap[c]) < 0) {
                temp = heap[r];
                heap[r] = heap[c];
                heap[c] = temp;
                r = c;
                c = 2 * c;
            } else {
                break;
            }
            c++;
        }


    }

    /**
     * Return the Order with the highest prep time from the queue without altering the queue
     *
     * @return the Order with the current longest prep time from the queue
     * @throws NoSuchElementException if the queue is empty
     */
    @Override
    public Order peekBest() {
        // TODO If the queue is empty, throw a NoSuchElementException
        if (this.isEmpty()) {
            throw new NoSuchElementException();
        }

        // TODO Return the Order with the longest prep time currently in the queue
        Order maxItem = queueHeap[0];
        return maxItem;
    }

    /**
     * Returns true if the queue contains no Orders, false otherwise
     *
     * @return true if the queue contains no Orders, false otherwise
     */
    @Override
    public boolean isEmpty() {
        // TODO implement this method according to its javadoc comment
        if (size == 0) {
            return true;
        }
        return false;
    }

    /**
     * Returns the number of elements currently in the queue
     *
     * @return the number of elements currently in the queue
     */
    public int size() {
        // TODO implement this method according to its javadoc comment
        return this.size;
    }

    /**
     * Creates a String representation of this PriorityQueue. Do not modify this implementation; this
     * is the version that will be used by all provided OrderPriorityQueue implementations that your
     * tester code will be run against.
     *
     * @return the String representation of this PriorityQueue, primarily for testing purposes
     */
    public String toString() {
        String toReturn = "";
        for (int i = 0; i < this.size; i++) {
            toReturn += queueHeap[i].getID() + "(" + queueHeap[i].getPrepTime() + ")";
            if (i < this.size - 1) toReturn += ", ";
        }
        return toReturn;
    }

}
