class CommentsController < ApplicationController

  before_action :set_comment, only: [:edit, :update, :destroy]

  def create
    @comment = Comment.new(comment_params)
    respond_to do |format|
      if @comment.save
        format.html { redirect_to blog_post_url(id: @comment.blog_post_id), notice: "Comment was successfully created!"}
      else
        format.html { redirect_to :back, notice: "Comment failed to post!"}
      end
    end
  end

  def update
    respond_to do |format|
      if @comment.update(comment_params)
        format.html { redirect_to blog_post_url(id: @comment.blog_post_id), notice: "Comment was updated successfully!"}
      else
        format.html { render :edit }
      end
    end
  end

  def destroy
    @comment.destory
    respond_to do |format|
      format.html { redirect_to blog_post_url(id: @comment.blog_post_id), notice: "Your comment was destroyed!"}
    end
  end


  def edit
  end

  private


    def comment_params
      params.require(:comment).permit(:user_id, :comment_entry, :blog_post_id)
    end

    def set_comment
      @comment = Comment.find(params[:id])
    end
end
