class Customer

  attr_accessor :name, :location

  def initialize(name, location)
    @name = name
    @location = location
  end

end

class Account

  attr_reader :acct_number, :balance
  attr_accessor :customer, :acct_type

  def initialize(acct_number, balance, acct_type, customer)
    @acct_number = acct_number
    @balance = balance
    @acct_type = acct_type
    @customer = customer
  end

  def deposit
    puts "How much money would you like to deposit?"
    amount = gets.chomp.to_f
    @balance += amount
    puts "Your new balance is $#{'%0.2f'%(@balance)}"
  end
  def withdrawl
    puts "How much money would you like to withdrawl?"
    amount = gets.chomp.to_f
    if @balance < amount
      @balance -= (amount + 25)
    else
      @balance -= amount
    end
    puts "Your new balance is $#{'%0.2f'%(@balance)}"
  end

end
