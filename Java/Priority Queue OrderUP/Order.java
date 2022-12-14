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

/**
 * Basic object representing a food order at this restaurant.
 * <p>
 * This class contains no mutator methods, only accessors.
 * <p>
 * TODO: implement the Comparable<Order> interface
 */
public class Order implements Comparable<Order> {

    private static int idGenerator = 1001;  // generator of unique order ID numbers

    private String dishName;                // name of the food associated with this Order
    private int prepTime;                   // approximate number of minutes to prepare this Order
    private final int ORDER_ID;             // unique order ID number

    /**
     * Constructor, initializes dish name and estimated prep time. Also sets this order's unique
     * identifier.
     *
     * @param dishName the name of the dish contained in this order
     * @param prepTime the approximate number of minutes required to prepare this dish
     */
    public Order(String dishName, int prepTime) {
        if (prepTime < 0) throw new IllegalArgumentException("Invalid prep time");
        this.ORDER_ID = idGenerator++;
        this.dishName = dishName;
        this.prepTime = prepTime;
    }

    // TODO: Implement the Comparable<Order> interface and any required methods such that two Orders
    // are equal if their prepTime values are equal, and an Order is "less than" another Order if its
    // prepTime is shorter than the other Order's.

    public int compareTo(Order obj) {

        if (obj instanceof Order) {

            Order veh = (Order) obj;
            if (this.prepTime == veh.prepTime)
                return 0;
            else if (this.prepTime < veh.prepTime)
                return -1;
        }
        return 1;
    }


    /**
     * Returns the name of the food associated with this Order
     *
     * @return the name of the food associated with this Order
     */
    public String getDishName() {
        return this.dishName;
    }

    /**
     * Returns the approximate number of minutes to prepare this Order
     *
     * @return the approximate number of minutes to prepare this Order
     */
    public int getPrepTime() {
        return this.prepTime;
    }

    /**
     * Returns the unique ID number of this Order
     *
     * @return the unique ID number of this Order
     */
    public int getID() {
        return this.ORDER_ID;
    }

    /**
     * Returns a String representation of this Order in the format "ID: dishname (prepTime)"
     *
     * @return a String representation of this Order
     */
    @Override
    public String toString() {
        return this.ORDER_ID + ": " + this.dishName + " (" + this.prepTime + ")";
    }

    /**
     * This method resets the idGenerator to 1001.
     * This method must be used in your tester methods only.
     */
    public static void resetIDGenerator() {
        idGenerator = 1001;
    }

}
