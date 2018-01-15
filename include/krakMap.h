#include <vector>
#include <set> // std::set
#include <deque> // std::deque
//#include <queue> // std::priority_queue
#include "sparsepp/spp.h"

#define NO_PARENT 0

// inclusive for begin 
// exclusive for end
struct Interval {
    uint32_t begin;
    uint32_t end;
    Interval() {}
    Interval(uint32_t inBegin, uint32_t inEnd) : begin(inBegin), end(inEnd) {}
};

enum class Rank : uint8_t {
    STRAIN, FORMA, VARIETAS, // strain sub group
    SUBSPECIES, SPECIES, SPECIESSUBGROUP, SPECIESGROUP, // species sub group 
    SUBGENUS, GENUS, SUPERGENUS, // genus sub group
    SUBTRIBE, TRIBE, SUBFAMILY, FAMILY, SUPERFAMILY, // family sub group
    PARVORDER, INFRAORDER, SUBORDER, ORDER, SUPERORDER, // order sub group
    COHORT, INFRACLASS, SUBCLASS, CLASS, SUPERCLASS, // class sub group
    SUBPHYLUM, PHYLUM, SUPERPHYLUM, // phylum sub group
    KINGDOM, SUPERKINGDOM , DOMAINE, LIFE // general sub group
};


class TaxaNode {
    public:
        TaxaNode() { 
            rank = Rank::STRAIN; 
            score = 0; 
            notIncorporatedChildrenCounter = 0;
        }
        TaxaNode(uint64_t inId, Rank inRank, uint64_t inPid) : 
            id(inId), score(0), parentId(inPid), 
            notIncorporatedChildrenCounter(0), 
            rank(inRank) {
                if (id == parentId) {
                    rank = Rank::LIFE;
                    parentId = NO_PARENT;
                }
            }
        bool isRoot() { return parentId == NO_PARENT; }//TODO not easy with the new design return children.size(); }
        bool isRipe() { return !notIncorporatedChildrenCounter;} // ripe if zero
        void addInterval(uint64_t begin, uint64_t len);
        void updateIntervals(TaxaNode* child);
        /**
         * Sorts intervals
         * Merge intervals if possible
         * Calculates score
         **/
        void cleanIntervalsAndCalcScore();
        bool addChild(TaxaNode* child);

        uint64_t getParentId() {return parentId;}
        uint64_t getId() {return id;}
        Rank getRank() {return rank;}
        uint64_t getScore() {return score;}
        std::set<uint64_t>& getActiveChildren() {return activeChildren;}
        std::vector<Interval>& getIntervals() {return intervals;}
        void reset();

    private:
        uint64_t id;
        uint32_t score;
        uint64_t parentId;
        uint64_t notIncorporatedChildrenCounter;
        std::set<uint64_t> activeChildren; //it's a set because, we might add the same child multiple times
        std::vector<Interval> intervals;
        Rank rank;
};


class KrakMap {
    public:
        KrakMap(std::string& taxonomyTree_filename, std::string& refId2TaxId_filename, std::string pruneLevelIn);
        bool classify(std::string& mapperOutput_filename);
        void saveClassificationResults(std::string& output_filename);
        void serialize(std::string& output_filename);
    private:
        void walk2theRoot(TaxaNode* child);
        void propagateInfo();
        void assignRead();
        void clearReadSubTree();
        void initializeRanks() {
            str2Rank["no rank"] = Rank::STRAIN;
            str2Rank["varietas"] = Rank::VARIETAS;
            str2Rank["subspecies"] = Rank::SUBSPECIES;
            str2Rank["species"] = Rank::SPECIES;
            str2Rank["species subgroup"] = Rank::SPECIESSUBGROUP;
            str2Rank["species group"] = Rank::SPECIESGROUP;
            str2Rank["subgenus"] = Rank::SUBGENUS;
            str2Rank["genus"] = Rank::GENUS;
            str2Rank["supergenus"] = Rank::SUPERGENUS;
            str2Rank["subfamily"] = Rank::SUBFAMILY;
            str2Rank["family"] = Rank::FAMILY;
            str2Rank["superfamily"] = Rank::SUPERFAMILY;
            str2Rank["subtribe"] = Rank::SUBTRIBE;
            str2Rank["tribe"] = Rank::TRIBE;
            str2Rank["forma"] = Rank::FORMA;
            str2Rank["cohort"] = Rank::COHORT;
            str2Rank["parvorder"] = Rank::PARVORDER;
            str2Rank["suborder"] = Rank::SUBORDER;
            str2Rank["order"] = Rank::ORDER;
            str2Rank["infraorder"] = Rank::INFRAORDER;
            str2Rank["superorder"] = Rank::SUPERORDER;
            str2Rank["subclass"] = Rank::SUBCLASS;
            str2Rank["class"] = Rank::CLASS;
            str2Rank["infraclass"] = Rank::INFRACLASS;
            str2Rank["superclass"] = Rank::SUPERCLASS;
            str2Rank["subphylum"] = Rank::SUBPHYLUM;
            str2Rank["phylum"] = Rank::PHYLUM;
            str2Rank["superphylum"] = Rank::SUPERPHYLUM;
            str2Rank["kingdom"] = Rank::KINGDOM;
            str2Rank["superkingdom"] = Rank::SUPERKINGDOM;
            str2Rank["domain"] = Rank::DOMAINE;
            str2Rank["life"] = Rank::LIFE; 
        }
        std::string rankToStr(Rank r) {
            switch (r) {
                case Rank::STRAIN:
                    return "no rank";
                case Rank::VARIETAS:
                    return "varietas";
                case Rank::SUBSPECIES:
                    return "subspecies";
                case Rank::SPECIES:
                    return "species";
                case Rank::SPECIESSUBGROUP:
                    return "species subgroup";
                case Rank::SPECIESGROUP:
                    return "species group";
                case Rank::SUBGENUS:
                    return "subgenus";
                case Rank::GENUS:
                    return "genus";
                case Rank::SUPERGENUS:
                    return "supergenus";
                case Rank::SUBFAMILY:
                    return "subfamily";
                case Rank::FAMILY:
                    return "family";
                case Rank::SUPERFAMILY:
                    return "superfamily";
                case Rank::SUBTRIBE:
                    return "subtribe";
                case Rank::TRIBE:
                    return "tribe";
                case Rank::FORMA:
                    return "forma";
                case Rank::COHORT:
                    return "cohort";
                case Rank::PARVORDER:
                    return "parvorder";
                case Rank::SUBORDER:
                    return "suborder";
                case Rank::ORDER:
                    return "order";
                case Rank::INFRAORDER:
                    return "infraorder";
                case Rank::SUPERORDER:
                    return "superorder";
                case Rank::SUBCLASS:
                    return "subclass";
                case Rank::CLASS:
                    return "class";
                case Rank::INFRACLASS:
                    return "infraclass";
                case Rank::SUPERCLASS:
                    return "superclass";
                case Rank::SUBPHYLUM:
                    return "subphylum";
                case Rank::PHYLUM:
                    return "phylum";
                case Rank::SUPERPHYLUM:
                    return "superphylum";
                case Rank::KINGDOM:
                    return "kingdom";
                case Rank::SUPERKINGDOM:
                    return "superkingdom";
                case Rank::DOMAINE:
                    return "domain";
                case Rank::LIFE:
                    return "life";
            }
            return "really!! none?";
        }
        spp::sparse_hash_map<uint32_t, TaxaNode> taxaNodeMap;
        spp::sparse_hash_map<std::string, uint32_t> refId2taxId;
        spp::sparse_hash_map<std::string, Rank> str2Rank;
        std::deque<TaxaNode*> hits;
        std::set<uint64_t> activeTaxa;
        Rank pruningLevel = Rank::SPECIES;
        TaxaNode* root = nullptr;
        spp::sparse_hash_map<uint64_t, std::pair<uint64_t, Rank>> mappedReadCntr;
    
};