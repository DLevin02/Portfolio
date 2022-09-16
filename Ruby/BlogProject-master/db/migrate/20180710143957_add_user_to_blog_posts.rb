class AddUserToBlogPosts < ActiveRecord::Migration[5.2]
  def change
    add_column :blog_posts, :user_id, :integer
  end
end
