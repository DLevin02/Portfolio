json.extract! pizza, :id, :name, :crust, :sauce, :topping, :cost, :image, :created_at, :updated_at
json.url pizza_url(pizza, format: :json)
