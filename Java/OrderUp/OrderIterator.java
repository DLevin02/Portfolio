import java.util.Iterator;
import java.util.NoSuchElementException;

public class OrderIterator implements Iterator<Order> {
    private LinkedOrder current;

    //Constructor, initializes current to the provided starting LinkedOrder.
    //Does not care whether the argument value is null.
    public OrderIterator(LinkedOrder start){
        this.current = start;

    }

    //○ Returns true if and only if the iteration has more orders
    public boolean hasNext(){
        if (this.current.getNext() != null) {
            return true;
        }
        return false;
    }

    //Throws a NoSuchElementException with a descriptive error message if the iteration does
    //not have more orders to return.
    //○ Otherwise returns the next Order and updates the current field appropriately.
    public Order next() throws NoSuchElementException{

        if(this.current.hasNext() == false){
            throw new NoSuchElementException("There are no more orders in the iteration");
        }else{
            this.current = this.current.getNext();
            return this.current;
        }

    }



}
