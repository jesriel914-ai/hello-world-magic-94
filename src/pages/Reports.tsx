import Layout from "@/components/Layout";
import PageWrapper from "@/components/PageWrapper";

const Reports = () => {
  return (
    <Layout>
      <PageWrapper skeletonType="table">
        <div className="px-6 py-4">
          <div>
            <h1 className="text-3xl font-bold text-education-navy">Reports</h1>
          </div>
        </div>
      </PageWrapper>
    </Layout>
  );
};

export default Reports;
