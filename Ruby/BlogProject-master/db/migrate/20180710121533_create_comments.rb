class CreateComments < ActiveRecord::Migration[5.2]
  def change
    create_table :comments do |t|
      t.string :author
      t.text :comment_entry
      t.integer :blog_post_id

      t.timestamps
    end
  end
end
