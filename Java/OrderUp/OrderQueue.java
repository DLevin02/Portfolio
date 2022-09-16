import java.util.Iterator;
import java.util.NoSuchElementException;

public class OrderQueue implements QueueADT<Order>, Iterable<Order>{

    private LinkedOrder front;
    private LinkedOrder back;
    private int size;

    //Creates and returns a new OrderIterator beginning with the current value of front
    public Iterator<Order> iterator(){
        return new OrderIterator<Order>(front);
    }

    //Adds a new LinkedOrder containing newElement to the back of the queue
    //size variable and front/back references appropriately
    public void enqueue(Order newElement){
        this.back = newElement;
        size++;

    }

    //Removes the next LinkedOrder from the front of the queue and returns its Order,
    //updating the size variable and front/back references appropriately
    //○ Throws a NoSuchElementException if the queue is empty
    public Order dequeue() throws NoSuchElementException{
        if(this.back == null){
            throw new NoSuchElementException("The Que is Empty");
        }else if(this.front.getNext() == null){
            throw new NoSuchElementException("The Que is Empty");
        }else{
            this.front = this.front.getNext();
            size--;
        }
    }

    //Returns the Order from the LinkedOrder at the front of the queue without removing the
    //LinkedOrder from the queue
    //Throws a NoSuchElementException if the queue is empty
    public Order peek() throws NoSuchElementException{
        if(this.front == null){
            throw new NoSuchElementException("The Que is Empty");
        }else{
            return this.front;
        }

    }

    //Returns true if and only if the queue is empty
    public boolean isEmpty(){
        if(this.front == null){
            return true;
        }else{
            return null;
        }

    }

    /**
     * Creates and returns a String representation of this OrderQueue
     * using an enhanced-for loop. For example, a queue with three Orders
     * might look like this:
     * 1001: fries (2) -> 1002: shake (1) -> 1003: burger (3) -> END
     *
     * @return A String representation of the queue
     */
    @Override
    public String toString() {
        if (this.size == 0) return "";
        String qString = "";
        for (Order o : this) {
            qString += o.toString();
            qString += " -> ";
        }
        qString += "END";
        return qString;
    }

}
