import pymongo


class MongoUtil:

    def __init__(self):
        pass

    @staticmethod
    def get_coll_creator_score():
        mongo_media_platfrom = pymongo.MongoClient(
            'mongodb://media_platform_20201221:q%21MpGRVY9jUiJEqJC3xj@media-platform.mongo.nb.com:27017/?authSource=media_platform&replicaSet=prod-media-platform-01&readPreference=secondaryPreferred&appname=MongoDB%20Compass&ssl=false')
        coll = mongo_media_platfrom['media_platform']['creator_score']
        return coll

    @staticmethod
    def get_coll_creator_level():
        mongo_media_platfrom = pymongo.MongoClient(
            'mongodb://media_platform_20201221:q%21MpGRVY9jUiJEqJC3xj@media-platform.mongo.nb.com:27017/?authSource=media_platform&replicaSet=prod-media-platform-01&readPreference=secondaryPreferred&appname=MongoDB%20Compass&ssl=false')
        coll = mongo_media_platfrom['media_platform']['creator_info']
        return coll

    @staticmethod
    def get_coll_creator_network_review_tasks():
        coll = pymongo.MongoClient("mongodb://creator_audit_20210419:ao1sw6hEQCK%2F45vv6@media-platform.mongo.nb.com",
                                   27017,
                                   unicode_decode_error_handler="ignore")["creator_audit"]["creator_network_review_tasks"]
        return coll

    @staticmethod
    def get_coll_static_feature():
        coll = pymongo.MongoClient("172.31.29.170", 27017, unicode_decode_error_handler="ignore")["staticFeature"][
            "document"]
        return coll
