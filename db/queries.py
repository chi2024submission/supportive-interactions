# get conversations: messages groupped by userid, threadid, convid
GET_CONVERSATIONS_TAGSET_0_IN_CHI_NHOT = '''select
	array_agg(s.line_text::TEXT order by line_num asc) as texts,
	array_agg(s.chi_gs_multi_nhot::SMALLINT[] order by line_num asc) as gs,
	array_agg(s.author::TEXT order by line_num asc) as author,
	user_id, thread_id, conv_id
from messenger s
where s.is_annotated_tagset_0 is true and user_id in %s
group by s.user_id, s.thread_id, s.conv_id;
'''

# get conversations: messages groupped by userid, threadid, convid
GET_EN_CONVERSATIONS_TAGSET_0_IN_CHI_NHOT = '''select
	array_agg(s.line_text_en::TEXT order by line_num asc) as texts,
	array_agg(s.chi_gs_multi_nhot::SMALLINT[] order by line_num asc) as gs,
	array_agg(s.author::TEXT order by line_num asc) as author,
	user_id, thread_id, conv_id
from messenger s
where s.is_annotated_tagset_0 is true and user_id in %s
group by s.user_id, s.thread_id, s.conv_id;
'''
