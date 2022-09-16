class RemoveAuthorFromBlogPost < ActiveRecord::Migration[5.2]
  def change
    remove_column :blog_posts, :author, :string
  end
end
