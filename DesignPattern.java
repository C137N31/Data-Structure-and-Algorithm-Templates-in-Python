// Singleton
// only one instance
public class ParkingLot {
    private ParkingLot() {}
    
    private static class SingletonParkingLot {
        static final ParkingLot _instance = new ParkingLot();
    }

    public static synchronized ParkingLot getInstance() {
        return SingletonParkingLot._instance;
    }
}

// State
// class methods will change state by using setState()
// different logics on different states
public class VendingMachine {
    private State state;
    private HasSelectionState hasSelectionState;
    private NoSelectionState noSelectionState;
    private SoldState soldState;
    private SoldOutState soldOutState;

    public void setState(State s) {
        state = s;
    }
}

public interface State {
    public float selectItem(String selection);
    public void insertCoins(List<Coin> coins);
    public Pair<Item, List<Coin>> executeTransaction();
    public List<Coin> cancelTransaction();
}

public class HasSelectionState implements State {
    VendingMachine vendingMachine = null;

    public HasSelectionState(VendingMachine machine) {
        vendingMachine = machine;
    }

    public float selectItem(String selection) {}
    public void insertCoins(List<Coin> coins) {}
    public Pair<Item, List<Coin>> executeTransaction() {}
    public List<Coin> cancelTransaction() {}
}

public class NoSelectionState implements State {
    VendingMachine vendingMachine = null;

    public NoSelectionState(VendingMachine machine) {
        vendingMachine = machine;
    }

    public float selectItem(String selection) {}
    public void insertCoins(List<Coin> coins) {}
    public Pair<Item, List<Coin>> executeTransaction() {}
    public List<Coin> cancelTransaction() {}
}

public class SoldState implements State {
    VendingMachine vendingMachine = null;

    public SoldState(VendingMachine machine) {
        vendingMachine = machine;
    }

    public float selectItem(String selection) {}
    public void insertCoins(List<Coin> coins) {}
    public Pair<Item, List<Coin>> executeTransaction() {}
    public List<Coin> cancelTransaction() {}
}

public class SoldOutState implements State {
    VendingMachine vendingMachine = null;

    public SoldOutState(VendingMachine machine) {
        vendingMachine = machine;
    }

    public float selectItem(String selection) {}
    public void insertCoins(List<Coin> coins) {}
    public Pair<Item, List<Coin>> executeTransaction() {}
    public List<Coin> cancelTransaction() {}
}

// Adapter
// convert coin's integer value to a string ItemName
public class CoinAdapter implements Item {
    private Coin coin;

    public CoinAdapter(Coin coin) {
        this.coin = coin;
    }

    public String getItemName() {
        return new String(coin.getValue());
    }
}

// Strategy
// behavior
public interface Strategy {
    public void processPayment(Payment payment);
}

public class PaypalStrategy implements Strategy {
    public void processPayment(Payment payment) {}
}

public class CreditCardStrategy implements Strategy {
    public void processPayment(Payment payment) {}
}

// Factory
// creation
public class StrategyFactory {
    public Strategy createStrategy(Payment payment){
        if (payment.getMethod().equals("paypal"))
            strategy = new PaypalStrategy();
        else if (payment.getMethod().equals("credit card"))
            strategy = new CreditCardStrategy();
    }

    public void pay(Payment payment) {
        strategy = createStrategy(payment);
        strategy.processPayment(payment);
    }
}

// Decorator
public interface Coffee {
    public double getCost();
    public String getIngredients();
}

public class SimpleCoffee implements Coffee {
    @Override
    public double getCost() {
        return 1;
    }

    @Override
    public String getIngredients() {
        return "Coffee";
    }
}

public abstract class CoffeeDecorator implements Coffee {
    protected final Coffee decoratedCoffee;

    public CoffeeDecorator(Coffee c) {
        this.decoratedCoffee = c;
    }

    public double getCost() {
        return decoratedCoffee.getCost();
    }

    public String getIngredients() {
        return decoratedCoffee.getIngredients();
    }
}

class WithMilk extends CoffeeDecorator {
    public WithMilk(Coffee c) {
        super(c);
    }

    public double getCost() {
        return super.getCost() + 0.5
    }

    public String getIngredients() {
        return super.getIngredients() + ", Milk";
    }
}

class WithSprinkles extends CoffeeDecorator {
    public WithSprinkles(Coffee c) {
        super(c);
    }

    public double getCost() {
        return super.getCost() + 0.2
    }

    public String getIngredients() {
        return super.getIngredients() + ", Sprinkles";
    }
}

public class CoffeeMaker {
    public static void printInfo(Coffee c) {
        System.out.println("Cost: " + c.getCost() + "; Ingredients: " + c.getIngredients());
    }

    public static void main(String[] args) {
        Coffee c = new SimpleCoffee();
        printInfo(c);

        c = new WithMilk(c);
        printInfo(c);

        c = new WithSprinkles(c);
        printInfo(c);
    }
}