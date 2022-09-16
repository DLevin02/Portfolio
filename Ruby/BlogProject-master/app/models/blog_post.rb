class BlogPost < ApplicationRecord
  has_many :comments, dependent: :destroy
  belongs_to :user
  validates :title, :blog_entry, presence: true
  validates :title, uniqueness: true, length: {maximum: 140}

  paginates_per 3
end
