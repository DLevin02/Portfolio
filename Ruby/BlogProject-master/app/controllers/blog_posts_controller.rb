class BlogPostsController < ApplicationController

  before_action :set_blog_post, only: [:show, :edit, :update, :destroy]
  before_action :authenticate_user!, except: [:index, :show]


  include ApplicationHelper

  def index
    @blog_posts = BlogPost.all.page(params[:page])
  end

  def show
    @comment = Comment.new
  end

  def user_posts
    @user = User.find_by(username: params[:name])
  end

  def edit
    no_access_visitors(current_user, @blog_post)
  end

  def new
    @blog_post = BlogPost.new #Creating a new instance of BlogPost (remember initialize from ruby classes)
  end

  def create
    @blog_post = BlogPost.new(blog_post_params) #method we will set up in a second here. This method will help us allow all of the BlogPost attributes
    respond_to do |format|
      if @blog_post.save #Method from the blogPost class
        	format.html { redirect_to blog_post_url(@blog_post), notice: "Blog Post
    		was created successfully."} #route when BlogPost successfully saved to database
      else
        format.html {render :new} #render to new blog post page when blog post can't be saved to database
      end
    end
  end

  def update
    respond_to do |format|
      if @blog_post.update(blog_post_params) #blog_post.update is from the BlogPost class
        format.html { redirect_to blog_post_url(@blog_post), notice: "Blog post was updated successfully"}
      else
        format.html { render :edit }
      end
    end
  end

  def destroy
    @blog_post.destroy #Method from the BlogPost class
    respond_to do |format|
      format.html { redirect_to blog_posts_url, notice: "Your Blog Post was DESTROYEDDDDDDDDDD!!!!!!!!!"}
      # Route to when a blog post is destroyed. I could have set instead also root_url as BlogPost index view is also the root page
    end
  end

  private

    def set_blog_post
        @blog_post = BlogPost.find(params[:id])
    end
    def blog_post_params
      params.require(:blog_post).permit(:title, :blog_entry, :user_id)
    end

end
