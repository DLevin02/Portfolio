module ApplicationHelper

  def no_access_visitors(current_user, post_user)
    unless current_user.id == post_user.user_id
      redirect_to root_url
    end
  end
end
