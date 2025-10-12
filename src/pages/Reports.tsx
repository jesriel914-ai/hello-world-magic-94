import Layout from "@/components/Layout";
import PageWrapper from "@/components/PageWrapper";

const Reports = () => {
  return (
    <Layout>
      <PageWrapper skeletonType="table">
        <div className="px-6 py-4">
          <div>
            <h1 className="text-2xl font-bold text-education-navy">REPORTS</h1>
          </div>
        </div>
      </PageWrapper>
    </Layout>
  );
};

export default Reports;
