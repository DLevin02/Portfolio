Rails.application.routes.draw do
  # For details on the DSL available within this file, see http://guides.rubyonrails.org/routing.html
  root 'weather_controller#index'
  get 'index' => 'weather_controller#index'

  post 'index' => 'weather_controller#index'

  get 'test' => 'weather_controller#test'
end
