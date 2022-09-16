// --== CS400 Project One File Header ==--
// Name: Drew Levin
// Email: dslevin2@wisc.edu
// Team: Blue
// Group: CO
// TA: Evan Wireman
// Lecturer: Florian
// Notes to Grader: <optional extra notes>


import java.util.LinkedList;
import java.util.NoSuchElementException;

/**
 * This class models a Box used for inventory
 *
 * @author Drew Levin
 *
 */
public class HashtableMap<KeyType, ValueType> implements MapADT<KeyType, ValueType> {


    /**
     * This class models a key value pair hashtable
     *
     * @author Drew Levin
     *
     */
    public class KeyValuePair{
        protected ValueType value;
        protected KeyType key;


        /**
         * Constructor for Key Value Pair
         *
         * @param key the key to be used in the instance
         * @param value the value associated with the key
         */
        public KeyValuePair(KeyType key, ValueType value){
            this.value = value;
            this.key = key;
        }

        /**
         * Constructor for Key Value Pair
         *
         * @param key the key to be used in the instance
         */
        public KeyValuePair(KeyType key){
            this.key = key;
        }
    }

    private int size;
    private int potential;
    private double load;
    private LinkedList<KeyValuePair>[] table;

    /**
     * Constructor for HashTableMap
     *
     * @param potential max capacity
     */
    public HashtableMap(int potential){
        this.potential = potential;
        this.size = 0;
        this.load = ((double)size)/(double)potential;
        table = (LinkedList<KeyValuePair>[]) new LinkedList[potential];
    }

    /**
     * Constructor for HashTableMap
     */
    public HashtableMap(){
        this.potential = 20;
        this.size=0;
        this.load = size/20.0;
        table = (LinkedList<KeyValuePair>[]) new LinkedList[20];
    }

    /**
     * Add Key Value Pair To Hash Table
     *
     * @param key Value of Key
     * @param value Value that will be added
     * @return False if value is not added or if it contains key otherwise True
     */
    @Override
    public boolean put(KeyType key, ValueType value) {

        if(containsKey(key)){
            return false;
        }

        KeyValuePair in = new KeyValuePair(key, value);
        int position = getPosition(key);

        if(this.table[position] == null){
            this.table[position] = new LinkedList<>();
        }

        this.table[position].add(in);
        this.size++;
        if(((double)this.size/(double)this.potential) >= .8){
            updateHash();
        }
        return true;
    }

    /**
     * Gets the value stored in the hash table at a given key
     *
     * @param key Key that will find value
     * @throws NoSuchElementException if not found in HashTable
     * @return value when the pair is found in the table
     */
    @Override
    public ValueType get(KeyType key) throws NoSuchElementException {


        ValueType value;
        int position = getPosition(key);


        if(table[position] == null){
            throw new NoSuchElementException("Element Does Not Exist");
        }
        for(int i = 0; i < table[position].size(); i++){
            if(table[position].get(i).key.equals(key)){
                value = table[position].get(i).value;
                return value;
            }
        }
        throw new NoSuchElementException("Element Does Not Exist");
    }

    /**
     * Returns the size of the table
     */
    @Override
    public int size() {
        return size;
    }

    /**
     * If hashTable contains key value pair it will return a certain value depending on validity
     *
     * @param key The key for the pair to be checked
     * @return true if the table does contain the key value pair
     */
    @Override
    public boolean containsKey(KeyType key) {


        int position = getPosition(key);

        if(table[position] == null){
            return false;
        }

        for(int i = 0; i<table[position].size(); i++){
            if(table[position].get(i).key.equals(key)){
                return true;
            }
        }
        return false;
    }

    /**
     * Removes the key value pair based on the parameter key that is given
     * @param key the key that is associated with the value pair to be removed
     * @return null of returnValue of the given key that is removed
     */
    @Override
    public ValueType remove(KeyType key) {

        int position = getPosition(key);
        if(table[position] == null) {
            return null;
        }

        if(!containsKey(key)){
            return null;
        }

        for(int i = 0; i<table[position].size(); i++){
            if(table[position].get(i).key.equals(key)){
                ValueType returnValue = table[position].get(i).value;
                table[position].remove(i);
                size = size - 1;
                return returnValue;
            }
        }

        return null;
    }

    /**
     * Clears the hash and completely resets it
     */
    @Override
    public void clear() {
        load = 0;
        size = 0;
        table = (LinkedList<KeyValuePair>[]) new LinkedList[potential];

    }

    /**
     * Grow by doubling its capacity and rehashing,
     * whenever its load factor becomes greater than or equal to 80%.
     */
    private void updateHash(){
        LinkedList<KeyValuePair>[] copyOfTable = this.table.clone();
        table = (LinkedList<KeyValuePair>[]) new LinkedList[potential*2];
        potential = potential * 2;
        load = 0.0;
        size = 0;

        for(int i = 0; i < copyOfTable.length; i++){
            if(copyOfTable[i]==null){
                continue;
            }
            for(int j = 0; j < copyOfTable[i].size(); j++){
                put(copyOfTable[i].get(j).key, copyOfTable[i].get(j).value);
            }
        }

    }

    /**
     * Gets the Index at a given key value
     * @param key The Key of the index to be found
     * @return the index value
     */
    private int getPosition(KeyType key){
        int hashCount = key.hashCode();
        if(hashCount < 0){
            hashCount = -hashCount;
        }
        return hashCount % potential;
    }

}
